# build_training_dataset.py
# 원본 + 증강 합치고 train/val/test 분할

import os
import csv
import json
import argparse
import random

from pipeline_config import (
    TRANSCRIPTION_DIR,
    AUGMENTED_DIR,
    FINAL_DIR,
    DEFAULT_VARIANT,
    CSV_COLUMNS,
    SPLIT_RATIOS,
    SPLIT_SEED,
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


def write_rows(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def split_filenames(filenames, ratios, seed):
    rng = random.Random(seed)
    fnames = list(filenames)
    rng.shuffle(fnames)

    n = len(fnames)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    train_files = set(fnames[:n_train])
    val_files = set(fnames[n_train:n_train + n_val])
    test_files = set(fnames[n_train + n_val:])

    return train_files, val_files, test_files


def main():
    parser = argparse.ArgumentParser(description="최종 학습 데이터셋 생성")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Whisper 변형")
    parser.add_argument("--input-llm", default=os.path.join(AUGMENTED_DIR, "llm_fewshot.csv"), help="LLM 증강 CSV")
    parser.add_argument("--input-asr", default=os.path.join(AUGMENTED_DIR, "asr_noised.csv"), help="ASR 노이즈 CSV")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="분할 시드")
    parser.add_argument("--output-dir", default=FINAL_DIR, help="출력 디렉터리")
    args = parser.parse_args()

    original_csv = os.path.join(TRANSCRIPTION_DIR, args.variant, "all.csv")
    original_rows = load_rows(original_csv)
    if not original_rows:
        raise SystemExit(f"원본 전사 CSV가 비어있음: {original_csv}")

    llm_rows = load_rows(args.input_llm)
    asr_rows = load_rows(args.input_asr)

    all_rows = original_rows + llm_rows + asr_rows

    filenames = sorted({row.get("filename", "") for row in original_rows if row.get("filename")})
    if not filenames:
        raise SystemExit("파일명 기반 분할을 위한 원본 filename이 없습니다.")

    train_files, val_files, test_files = split_filenames(filenames, SPLIT_RATIOS, args.seed)

    train_rows, val_rows, test_rows = [], [], []
    for row in all_rows:
        fname = row.get("filename", "")
        if fname in train_files:
            train_rows.append(row)
        elif fname in val_files:
            val_rows.append(row)
        elif fname in test_files:
            test_rows.append(row)
        else:
            train_rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    val_path = os.path.join(args.output_dir, "val.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    write_rows(train_path, train_rows)
    write_rows(val_path, val_rows)
    write_rows(test_path, test_rows)

    def count_by(rows, key):
        out = {}
        for r in rows:
            k = str(r.get(key, ""))
            out[k] = out.get(k, 0) + 1
        return out

    stats = {
        "total_rows": len(all_rows),
        "unique_filenames": len(filenames),
        "splits": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "label_distribution": count_by(all_rows, "label"),
        "source_distribution": count_by(all_rows, "source"),
        "augmented_ratio": (len(all_rows) - len(original_rows)) / len(all_rows),
    }

    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"train: {train_path} ({len(train_rows)}건)")
    print(f"val:   {val_path} ({len(val_rows)}건)")
    print(f"test:  {test_path} ({len(test_rows)}건)")
    print(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
