"""
generate_normal_combined.py

minsang_train.csv (Track A, 상담 대화) 와 normal_scripts.csv (Track B, 일상 대화) 를
합쳐 normal_combined.csv 를 생성합니다.

화자 레이블(고객:, 상담사:, 1 :, 2 : 등)과 줄바꿈을 제거하여
LLM 증강 데이터와 동일한 포맷(화자 구분 없는 연속 텍스트)으로 정규화합니다.
이 정규화는 ASR 노이즈 주입 전에 적용되어야 합니다.

출력 컬럼: id, script, label, track
  - track A: minsang_train.csv (상담 대화, label=0)
  - track B: normal_scripts.csv (일상 대화, label=0)
"""

import argparse
import csv
import re
from pathlib import Path

BASE_DIR          = Path(__file__).resolve().parent
PREPROCESSING_DIR = BASE_DIR.parent
TRANSCRIPTION_DIR = PREPROCESSING_DIR / "transcriptions" / "normal_clean"
OUT_DIR           = BASE_DIR / "output"

MINSANG_CSV       = TRANSCRIPTION_DIR / "minsang_train.csv"
NORMAL_SCRIPTS_CSV = TRANSCRIPTION_DIR / "normal_scripts.csv"
OUTPUT_CSV        = OUT_DIR / "normal_combined.csv"

# 화자 레이블 패턴: "고객:", "상담사:", "1 :", "2 :" 등 (앞뒤 공백 허용)
_SPEAKER_RE = re.compile(r"^(?:[가-힣]+|\d+)\s*:\s*")


def normalize_speaker_text(text: str) -> str:
    """화자 레이블 및 줄바꿈을 제거하고 연속 텍스트로 변환.

    입력 예:
      "고객: 안녕하세요\n상담사: 네, 안녕하세요"
      "1 : 너 옷 어디서 사?\n2 : 주로 인터넷으로"
    출력 예:
      "안녕하세요 네 안녕하세요"
      "너 옷 어디서 사 주로 인터넷으로"
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = _SPEAKER_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    return " ".join(cleaned)


def load_minsang(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw_id = str(r.get("id", "")).strip()
            script = normalize_speaker_text(r.get("script", "") or "")
            if not script:
                continue
            rows.append({
                "id":     f"minsang_{raw_id}",
                "script": script,
                "label":  "0",
                "track":  "A",
            })
    return rows


def load_normal_scripts(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw_id = str(r.get("id", "")).strip()
            script = normalize_speaker_text(r.get("script", "") or "")
            if not script:
                continue
            rows.append({
                "id":     f"normal_{raw_id}",
                "script": script,
                "label":  "0",
                "track":  "B",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="normal_combined.csv 생성 (화자 레이블 정규화 포함)")
    parser.add_argument("--minsang",        default=str(MINSANG_CSV))
    parser.add_argument("--normal-scripts", default=str(NORMAL_SCRIPTS_CSV))
    parser.add_argument("--output",         default=str(OUTPUT_CSV))
    args = parser.parse_args()

    minsang_rows = load_minsang(Path(args.minsang))
    normal_rows  = load_normal_scripts(Path(args.normal_scripts))
    all_rows     = minsang_rows + normal_rows

    by_track = {}
    for r in all_rows:
        t = r["track"]
        by_track[t] = by_track.get(t, 0) + 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "script", "label", "track"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[DONE] {args.output}")
    print(f"  Track A (minsang): {by_track.get('A', 0)}건")
    print(f"  Track B (normal):  {by_track.get('B', 0)}건")
    print(f"  합계:              {len(all_rows)}건")


if __name__ == "__main__":
    main()
