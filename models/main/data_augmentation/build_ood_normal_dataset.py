"""
build_ood_normal_dataset.py

학습/검증/테스트에 사용되지 않은 일반 대화 원본 데이터로 OOD 테스트셋 생성.

출처:
  - 상담 대화 (label 0): transcriptions/normal_clean/minsang_train.csv
                         transcriptions/normal_clean/minsang_val.csv
  - 일상 대화 (label 1): transcriptions/normal_clean/normal_scripts.csv

출력: output/4class/ood.csv  (기존 train/val/test.csv 와 동일한 컬럼)
"""

import argparse
import csv
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MINSANG_TRAIN_CSV   = BASE_DIR / "transcriptions" / "normal_clean" / "minsang_train.csv"
MINSANG_VAL_CSV     = BASE_DIR / "transcriptions" / "normal_clean" / "minsang_val.csv"
NORMAL_SCRIPTS_CSV  = BASE_DIR / "transcriptions" / "normal_clean" / "normal_scripts.csv"
SPLIT_DIR           = BASE_DIR / "output" / "4class"
OUTPUT_CSV          = SPLIT_DIR / "ood.csv"

CSV_COLUMNS = ["id", "text", "label", "binary_label", "category", "source", "filename", "segment_risks"]

_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")


def clean_text(text: str) -> str:
    cleaned = _PUNCT_RE.sub("", text)
    cleaned = cleaned.replace("\n", " ")
    return re.sub(r" +", " ", cleaned).strip()


def load_used_ids() -> set[str]:
    """학습/검증/테스트에 사용된 normal_asr_orig id 수집."""
    used: set[str] = set()
    for split in ["train", "val", "test"]:
        path = SPLIT_DIR / f"{split}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("source") == "normal_asr_orig":
                    used.add(row["id"])
    return used


def load_minsang(used_ids: set[str]) -> list[dict]:
    rows = []
    for csv_path in [MINSANG_TRAIN_CSV, MINSANG_VAL_CSV]:
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ood_id = f"asr_minsang_{row['id']}"
                if ood_id in used_ids:
                    continue
                text = clean_text((row.get("script") or "").strip())
                if not text:
                    continue
                rows.append({
                    "id":            ood_id,
                    "text":          text,
                    "label":         0,
                    "binary_label":  0,
                    "category":      "상담 대화",
                    "source":        "ood_minsang",
                    "filename":      row.get("file_name", ""),
                    "segment_risks": "[]",
                })
    return rows


def load_normal_scripts(used_ids: set[str]) -> list[dict]:
    rows = []
    if not NORMAL_SCRIPTS_CSV.exists():
        return rows
    with NORMAL_SCRIPTS_CSV.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ood_id = f"asr_normal_{row['id']}"
            if ood_id in used_ids:
                continue
            text = clean_text((row.get("script") or "").strip())
            if not text:
                continue
            rows.append({
                "id":            ood_id,
                "text":          text,
                "label":         1,
                "binary_label":  0,
                "category":      "일상 대화",
                "source":        "ood_normal_scripts",
                "filename":      row.get("file_name", ""),
                "segment_risks": "[]",
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-consult", type=int, default=500, help="상담 대화 샘플 수 (default: 500)")
    parser.add_argument("--n-casual",  type=int, default=500, help="일상 대화 샘플 수 (default: 500)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default=str(OUTPUT_CSV))
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("[INFO] 사용된 id 로드 중...")
    used_ids = load_used_ids()
    print(f"  사용된 normal_asr_orig id: {len(used_ids)}건")

    print("[INFO] minsang 로드 중 (상담 대화)...")
    consult_pool = load_minsang(used_ids)
    print(f"  미사용 minsang: {len(consult_pool)}건")

    print("[INFO] normal_scripts 로드 중 (일상 대화)...")
    casual_pool = load_normal_scripts(used_ids)
    print(f"  미사용 normal_scripts: {len(casual_pool)}건")

    rng.shuffle(consult_pool)
    rng.shuffle(casual_pool)

    selected_consult = consult_pool[:args.n_consult]
    selected_casual  = casual_pool[:args.n_casual]

    if len(selected_consult) < args.n_consult:
        print(f"  [경고] 상담 대화 요청 {args.n_consult}건 중 {len(selected_consult)}건만 확보")
    if len(selected_casual) < args.n_casual:
        print(f"  [경고] 일상 대화 요청 {args.n_casual}건 중 {len(selected_casual)}건만 확보")

    all_rows = selected_consult + selected_casual
    rng.shuffle(all_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[DONE] {out_path}")
    print(f"  상담 대화: {len(selected_consult)}건")
    print(f"  일상 대화: {len(selected_casual)}건")
    print(f"  합계:      {len(all_rows)}건")


if __name__ == "__main__":
    main()
