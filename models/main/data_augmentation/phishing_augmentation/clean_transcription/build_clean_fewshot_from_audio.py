import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from pipeline_config import DEFAULT_VARIANT, TRANSCRIPTION_DIR


def normalize_category_dir_name(name: str) -> str:
    return (
        name.replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .strip()
    )


def find_audio_map(audio_root: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg", "*.webm"):
        for p in Path(audio_root).rglob(ext):
            mapping[p.name].append(str(p))
    return mapping


def build_category_dirs(audio_root: str) -> dict[str, str]:
    out: dict[str, str] = {}
    root = Path(audio_root)
    for p in root.iterdir():
        if p.is_dir():
            out[normalize_category_dir_name(p.name)] = str(p)
    return out


def resolve_audio_path(
    row_category: str,
    filename: str,
    audio_map: dict[str, list[str]],
    category_dirs: dict[str, str],
) -> tuple[str, str]:
    cat_key = normalize_category_dir_name(row_category)
    category_dir = category_dirs.get(cat_key)
    if category_dir:
        candidate = os.path.join(category_dir, filename)
        if os.path.exists(candidate):
            return candidate, ""

    paths = audio_map.get(filename, [])
    if not paths:
        return "", "missing"
    if len(paths) > 1:
        return "", "ambiguous"
    return paths[0], ""


def load_candidates(phishing_csv: str, min_len: int, max_len: int) -> list[dict]:
    rows = []
    with open(phishing_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if min_len <= len(text) <= max_len:
                rows.append(row)
    return rows


def transcribe_audio(client: OpenAI, audio_path: str, model: str, language: str) -> str:
    with open(audio_path, "rb") as af:
        resp = client.audio.transcriptions.create(
            model=model,
            file=af,
            language=language,
            response_format="text",
        )
    if isinstance(resp, str):
        return resp.strip()
    return (getattr(resp, "text", "") or "").strip()


def main():
    parser = argparse.ArgumentParser(description="오디오 재전사 기반 clean few-shot 생성")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--audio-root", default="models/classifier/data/phishing")
    parser.add_argument("--model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--min-len", type=int, default=1000)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--per-category", type=int, default=30)
    parser.add_argument("--loan-count", type=int, default=30, help="대출 사기형 추출 건수")
    parser.add_argument("--investigation-count", type=int, default=15, help="수사기관 사칭형(원클래스) 추출 건수")
    parser.add_argument("--voice-count", type=int, default=15, help="바로 이 목소리(원클래스) 추출 건수")
    parser.add_argument(
        "--output-csv",
        default="models/classifier/preprocessing/phishing_augmentation/output/phishing_clean_fewshot.csv",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    phishing_csv = os.path.join(TRANSCRIPTION_DIR, args.variant, "phishing.csv")
    candidates = load_candidates(phishing_csv, args.min_len, args.max_len)
    audio_map = find_audio_map(args.audio_root)
    category_dirs = build_category_dirs(args.audio_root)
    client = OpenAI(api_key=api_key)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_cat[(row.get("category") or "").strip()].append(row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    written = 0
    missing_audio = 0
    ambiguous_audio = 0

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "id",
            "category",
            "source_group",
            "filename",
            "audio_path",
            "text_clean",
            "source_model",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # 피싱 2클래스 운영 기준:
        # - 대출 사기형: 30
        # - 수사기관 사칭형: 15 (수사기관 폴더) + 15 (바로 이 목소리 폴더)
        quotas = {
            "대출 사기형": args.loan_count,
            "수사기관 사칭형": args.investigation_count,
            "바로 이 목소리": args.voice_count,
        }

        rows_by_group: dict[str, list[dict]] = defaultdict(list)
        seen_keys = set()
        for category, rows in by_cat.items():
            for row in rows:
                filename = (row.get("filename") or "").strip()
                if not filename:
                    continue
                audio_path, reason = resolve_audio_path(category, filename, audio_map, category_dirs)
                if reason == "missing":
                    missing_audio += 1
                    continue
                if reason == "ambiguous":
                    ambiguous_audio += 1
                    continue
                source_group = Path(audio_path).parent.name.strip()
                key = (source_group, filename)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                copied = dict(row)
                copied["_audio_path"] = audio_path
                copied["_source_group"] = source_group
                rows_by_group[source_group].append(copied)

        for source_group, quota in quotas.items():
            rows = rows_by_group.get(source_group, [])
            count = 0
            for row in rows:
                if count >= quota:
                    break
                audio_path = row["_audio_path"]
                filename = (row.get("filename") or "").strip()

                text_clean = transcribe_audio(client, audio_path, args.model, args.language)
                if not text_clean:
                    continue

                final_category = "수사기관 사칭형" if source_group in ("수사기관 사칭형", "바로 이 목소리") else "대출 사기형"

                writer.writerow(
                    {
                        "id": row.get("id", ""),
                        "category": final_category,
                        "source_group": source_group,
                        "filename": filename,
                        "audio_path": audio_path,
                        "text_clean": text_clean,
                        "source_model": args.model,
                    }
                )
                count += 1
                written += 1
            print(f"[OK] {source_group}: {count}건")

    print(f"[DONE] {args.output_csv} (총 {written}건)")
    print(f"[INFO] audio 없음: {missing_audio}건, filename 중복경로(스킵): {ambiguous_audio}건")


if __name__ == "__main__":
    main()
