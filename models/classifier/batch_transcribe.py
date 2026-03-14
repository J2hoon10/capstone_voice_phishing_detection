# batch_transcribe.py
# 전체 음성 데이터 일괄 STT 처리 (이력 기반 resume 지원)
#
# 사용법:
#   python batch_transcribe.py                    # 기본(gpu_small)
#   python batch_transcribe.py --variant cpu_base # CPU 모드
#   python batch_transcribe.py --resume           # 중단 지점부터 재개

import os
import sys
import csv
import argparse
from datetime import datetime
from collections import defaultdict

from pipeline_config import (
    DATA_DIR, TRANSCRIPTION_DIR, AUDIO_EXTENSIONS,
    WHISPER_VARIANTS, DEFAULT_VARIANT, CSV_COLUMNS,
    PHISHING_DIR, NORMAL_DIR,
)
from audio_processor import AudioProcessor

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _progress_iter(items, total, desc, enable):
    if tqdm and enable:
        return tqdm(items, total=total, desc=desc, unit="file")
    return items


def _numeric_filename_key(filename):
    stem, _ = os.path.splitext(filename)
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())


def collect_audio_files(base_dir):
    """하위 폴더 재귀 탐색하여 오디오 파일 목록 수집"""
    files = []
    for root, _, filenames in os.walk(base_dir):
        for f in sorted(filenames, key=_numeric_filename_key):
            if f.lower().endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(root, f))
    return files


def load_completed_by_category(csv_path):
    """이미 처리된 파일명 집합 반환 (resume용, category 기준 분리)"""
    completed = defaultdict(set)
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed[row["category"]].add(row["filename"])
    return completed


def count_rows(csv_path):
    """CSV 데이터 행 수 반환 (resume/summary 표시용)"""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def transcribe_category(processor, audio_files, label, category, output_csv, completed, show_progress):
    """한 카테고리의 오디오 파일들을 STT 처리하여 CSV에 append"""
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    mode = "a" if file_exists else "w"

    with open(output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        skipped = 0
        remaining = []
        for filepath in audio_files:
            fname = os.path.basename(filepath)
            if fname in completed:
                skipped += 1
            else:
                remaining.append(filepath)

        iterator = _progress_iter(
            remaining,
            total=len(remaining),
            desc=f"{label}/{category}",
            enable=show_progress,
        )

        for i, filepath in enumerate(iterator, 1):
            fname = os.path.basename(filepath)
            if tqdm and show_progress:
                iterator.set_postfix_str(fname)

            sentences = processor.process_file(filepath)
            text = " ".join(sentences) if sentences else ""

            row = {
                "id": f"{label}_{category}_{os.path.splitext(fname)[0]}",
                "text": text,
                "label": 1 if label == "phishing" else 0,
                "category": category,
                "source": "original",
                "filename": fname,
            }
            writer.writerow(row)
            completed.add(fname)

        if skipped > 0:
            print(f"  (이미 처리된 {skipped}개 건너뜀)")


def merge_csvs(output_dir, label_csvs):
    """phishing.csv + normal.csv → all.csv 합본"""
    all_csv = os.path.join(output_dir, "all.csv")
    rows = []
    for csv_path in label_csvs:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))

    with open(all_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n합본 저장: {all_csv} ({len(rows)}건)")


def main():
    parser = argparse.ArgumentParser(description="음성 데이터 일괄 STT 처리")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=list(WHISPER_VARIANTS.keys()),
                        help="Whisper 모델 변형 선택")
    parser.add_argument("--resume", action="store_true",
                        help="이전 처리 이력 기반으로 중단 지점부터 재개")
    parser.add_argument("--no-progress", action="store_true",
                        help="진행률 표시 비활성화")
    args = parser.parse_args()

    variant = WHISPER_VARIANTS[args.variant]
    output_dir = os.path.join(TRANSCRIPTION_DIR, args.variant)
    os.makedirs(output_dir, exist_ok=True)

    phishing_csv = os.path.join(output_dir, "phishing.csv")
    normal_csv = os.path.join(output_dir, "normal.csv")

    # resume 모드: 기존 CSV에서 처리 완료된 파일 로드
    completed_phishing = load_completed_by_category(phishing_csv) if args.resume else defaultdict(set)
    completed_normal = load_completed_by_category(normal_csv) if args.resume else defaultdict(set)

    print(f"=== 일괄 STT 처리 ===")
    print(f"Whisper: {variant['model']} ({variant['device']}, {variant['compute_type']})")
    print(f"출력: {output_dir}")
    if args.resume:
        print(f"Resume: 피싱 {count_rows(phishing_csv)}건, 일반 {count_rows(normal_csv)}건 완료됨")
    print()
    if args.no_progress or not tqdm:
        if not tqdm:
            print("[INFO] tqdm 미설치: 진행률 바 없이 실행합니다.")
        else:
            print("[INFO] 진행률 표시 비활성화.")

    # AudioProcessor 초기화
    print("AudioProcessor 로딩 중...")
    processor = AudioProcessor(
        whisper_model_size=variant["model"],
        device=variant["device"],
        compute_type=variant["compute_type"],
    )
    print("로딩 완료!\n")

    # 피싱 데이터 처리
    if os.path.exists(PHISHING_DIR):
        categories = sorted([d for d in os.listdir(PHISHING_DIR)
                            if os.path.isdir(os.path.join(PHISHING_DIR, d))])
        for cat in categories:
            cat_dir = os.path.join(PHISHING_DIR, cat)
            files = collect_audio_files(cat_dir)
            print(f"\n[피싱/{cat}] {len(files)}개 파일")
            completed_set = completed_phishing[cat]
            transcribe_category(
                processor, files, "phishing", cat, phishing_csv, completed_set,
                show_progress=not args.no_progress,
            )

    # 일반 데이터 처리
    if os.path.exists(NORMAL_DIR):
        categories = sorted([d for d in os.listdir(NORMAL_DIR)
                            if os.path.isdir(os.path.join(NORMAL_DIR, d))])
        for cat in categories:
            cat_dir = os.path.join(NORMAL_DIR, cat)
            files = collect_audio_files(cat_dir)
            print(f"\n[일반/{cat}] {len(files)}개 파일")
            completed_set = completed_normal[cat]
            transcribe_category(
                processor, files, "normal", cat, normal_csv, completed_set,
                show_progress=not args.no_progress,
            )

    # 합본 생성
    merge_csvs(output_dir, [phishing_csv, normal_csv])

    # 요약
    p_count = count_rows(phishing_csv)
    n_count = count_rows(normal_csv)
    print(f"\n=== 완료 ===")
    print(f"피싱: {p_count}건, 일반: {n_count}건, 총: {p_count + n_count}건")
    print(f"결과: {output_dir}")


if __name__ == "__main__":
    main()
