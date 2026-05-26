# sample_ground_truth.py
# 오류 측정용 샘플 추출 및 어노테이션 템플릿 생성

import os
import csv
import argparse
import random

from pipeline_config import (
    TRANSCRIPTION_DIR,
    ERROR_ANALYSIS_DIR,
    DEFAULT_VARIANT,
    GROUND_TRUTH_SAMPLE_COUNT,
    GROUND_TRUTH_SEED,
)


def load_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_category(rows):
    grouped = {}
    for row in rows:
        cat = (row.get("category") or "unknown").strip()
        grouped.setdefault(cat, []).append(row)
    return grouped


def sample_per_category(grouped, count_per_category, seed):
    rng = random.Random(seed)
    sampled = []
    for cat, items in sorted(grouped.items(), key=lambda x: x[0]):
        rng.shuffle(items)
        take = items[:count_per_category]
        sampled.extend(take)
        if len(items) < count_per_category:
            print(f"[WARN] 카테고리 '{cat}' 샘플 부족: {len(items)}개만 사용")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="오류 측정용 샘플 추출")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Whisper 변형")
    parser.add_argument("--label", default="phishing", choices=["phishing", "normal", "all"], help="대상 라벨")
    parser.add_argument("--count", type=int, default=GROUND_TRUTH_SAMPLE_COUNT, help="카테고리당 샘플 수")
    parser.add_argument("--seed", type=int, default=GROUND_TRUTH_SEED, help="랜덤 시드")
    parser.add_argument("--input", default=None, help="입력 CSV 경로 (기본: variant 기반)")
    parser.add_argument("--output", default=None, help="출력 템플릿 CSV 경로")
    args = parser.parse_args()

    if args.input:
        input_csv = args.input
    else:
        if args.label == "phishing":
            input_csv = os.path.join(TRANSCRIPTION_DIR, args.variant, "phishing.csv")
        elif args.label == "normal":
            input_csv = os.path.join(TRANSCRIPTION_DIR, args.variant, "normal.csv")
        else:
            input_csv = os.path.join(TRANSCRIPTION_DIR, args.variant, "all.csv")

    output_csv = args.output or os.path.join(ERROR_ANALYSIS_DIR, "ground_truth_template.csv")

    rows = load_rows(input_csv)
    if not rows:
        raise SystemExit(f"입력 CSV를 찾을 수 없거나 비어있음: {input_csv}")

    grouped = group_by_category(rows)
    sampled = sample_per_category(grouped, args.count, args.seed)

    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["id", "filename", "label", "category", "stt_text", "human_transcription"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sampled:
            writer.writerow({
                "id": row.get("id", ""),
                "filename": row.get("filename", ""),
                "label": row.get("label", ""),
                "category": row.get("category", ""),
                "stt_text": row.get("text", ""),
                "human_transcription": "",
            })

    print(f"샘플 {len(sampled)}건 저장: {output_csv}")


if __name__ == "__main__":
    main()
