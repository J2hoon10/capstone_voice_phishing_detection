# batch_transcribe.py
# 전체 음성 데이터 일괄 STT 처리 (이력 기반 resume 지원)
# GPU 최적화: BatchedInferencePipeline + I/O 프리페치
#
# 사용법:
#   python batch_transcribe.py                    # 기본(gpu_small)
#   python batch_transcribe.py --variant cpu_base # CPU 모드
#   python batch_transcribe.py --resume           # 중단 지점부터 재개

import os
import sys

# audio_processor.py는 부모 디렉터리(models/classifier/)에 있음
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from pipeline_config import (
    TRANSCRIPTION_DIR, AUDIO_EXTENSIONS,
    WHISPER_VARIANTS, DEFAULT_VARIANT, CSV_COLUMNS,
    PHISHING_DIR, NORMAL_DIR,
    BATCH_SIZE,
)
from audio_enhancer import AudioEnhancer
from faster_whisper import WhisperModel, BatchedInferencePipeline

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _progress_iter(items, total, desc, enable):
    if tqdm and enable:
        return tqdm(items, total=total, desc=desc, unit="file")
    return items


def _is_cuda_missing_error(err):
    msg = str(err)
    return ("cublas64_12.dll" in msg) or ("CUDA" in msg and "not found" in msg)


def _numeric_filename_key(filename):
    stem, _ = os.path.splitext(filename)
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.lower())



def _init_pipeline(variant):
    whisper_model = WhisperModel(
        variant["model"],
        device=variant["device"],
        compute_type=variant["compute_type"],
    )
    return BatchedInferencePipeline(model=whisper_model)


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


def transcribe_category(state, enhancer, audio_files, label, category,
                        output_csv, completed, batch_size, show_progress):
    """한 카테고리의 오디오 파일들을 STT 처리하여 CSV에 append

    최적화:
    - BatchedInferencePipeline으로 STT 배치 추론
    - ThreadPoolExecutor로 다음 파일 오디오 로딩을 현재 STT와 병렬화
    """
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    mode = "a" if file_exists else "w"

    # 처리할 파일만 필터링
    pending = []
    skipped = 0
    for filepath in audio_files:
        fname = os.path.basename(filepath)
        if fname in completed:
            skipped += 1
        else:
            pending.append(filepath)

    if skipped > 0:
        print(f"  (이미 처리된 {skipped}개 건너뜀)")

    if not pending:
        return

    with open(output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        # I/O 프리페치: 다음 파일 1개만 백그라운드에서 미리 로딩
        with ThreadPoolExecutor(max_workers=1) as executor:
            iterator = _progress_iter(
                pending,
                total=len(pending),
                desc=f"{label}/{category}",
                enable=show_progress,
            )

            # 첫 번째 파일 미리 제출
            next_future = executor.submit(enhancer.enhance, pending[0]) if pending else None

            for i, filepath in enumerate(iterator, 1):
                fname = os.path.basename(filepath)
                if tqdm and show_progress:
                    iterator.set_postfix_str(fname)

                # 현재 파일의 enhance 결과 가져오기
                current_future = next_future

                # 다음 파일 미리 제출 (현재 STT와 병렬로 로딩)
                if i < len(pending):
                    next_future = executor.submit(enhancer.enhance, pending[i])
                else:
                    next_future = None

                enhanced_audio = current_future.result()

                # Whisper는 파일 경로 또는 numpy array를 받음
                # enhance()가 numpy array를 반환하므로 그대로 사용 가능
                # None이면 원본 파일 경로로 fallback
                if enhanced_audio is None:
                    target_input = filepath  # fallback: 원본 파일 경로
                elif isinstance(enhanced_audio, np.ndarray):
                    target_input = enhanced_audio  # numpy array 직접 전달
                else:
                    target_input = filepath

                # BatchedInferencePipeline STT
                import time
                t_start = time.time()

                def _do_transcribe(pipeline):
                    segs, info = pipeline.transcribe(
                        target_input,
                        language="ko",
                        batch_size=batch_size,
                        beam_size=1,
                        vad_filter=True,
                        no_speech_threshold=0.85,        # 0.6→0.85: 환각 반복 억제
                        condition_on_previous_text=False, # 이전 텍스트 문맥 비사용: 반복 환각 방지
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        compression_ratio_threshold=2.2,
                        log_prob_threshold=-0.8,
                        temperature=(0.0, 0.2, 0.4, 0.6),
                    )
                    # generator를 즉시 소비해야 CUDA 오류가 여기서 발생
                    return " ".join([seg.text for seg in segs])

                try:
                    text = _do_transcribe(state["pipeline"])
                except RuntimeError as e:
                    if state["variant"]["device"] == "cuda" and _is_cuda_missing_error(e):
                        print("[WARN] CUDA 라이브러리를 찾지 못해 CPU로 자동 전환합니다.")
                        cpu_variant = WHISPER_VARIANTS["cpu_base"]
                        state["pipeline"] = _init_pipeline(cpu_variant)
                        state["variant"] = cpu_variant
                        text = _do_transcribe(state["pipeline"])
                    else:
                        raise

                t_end = time.time()
                if not show_progress:
                    print(f"   [Profile] STT: {(t_end - t_start):.4f}s")

                row = {
                    "id": f"{label}_{category}_{os.path.splitext(fname)[0]}",
                    "text": text,
                    "label": 1 if label == "phishing" else 0,
                    "category": category,
                    "source": "original",
                    "filename": fname,
                }
                writer.writerow(row)
                f.flush()  # 파일별로 즉시 기록 (중단 시 데이터 보존)
                completed.add(fname)


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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"STT 배치 크기 (기본: {BATCH_SIZE})")
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

    print(f"=== 일괄 STT 처리 (GPU 최적화) ===")
    print(f"Whisper: {variant['model']} ({variant['device']}, {variant['compute_type']})")
    print(f"배치 크기: {args.batch_size}")
    print(f"출력: {output_dir}")
    if args.resume:
        print(f"Resume: 피싱 {count_rows(phishing_csv)}건, 일반 {count_rows(normal_csv)}건 완료됨")
    print()
    if args.no_progress or not tqdm:
        if not tqdm:
            print("[INFO] tqdm 미설치: 진행률 바 없이 실행합니다.")
        else:
            print("[INFO] 진행률 표시 비활성화.")

    # AudioEnhancer 초기화 (GPU noisereduce 포함)
    print("AudioEnhancer 로딩 중...")
    enhancer = AudioEnhancer(enable_vad=False)
    print("AudioEnhancer 로딩 완료!\n")

    # Whisper + BatchedInferencePipeline 초기화
    print("Whisper + BatchedInferencePipeline 로딩 중...")
    state = {"variant": variant, "pipeline": None}
    try:
        state["pipeline"] = _init_pipeline(variant)
    except RuntimeError as e:
        if variant["device"] == "cuda" and _is_cuda_missing_error(e):
            print("[WARN] CUDA 라이브러리를 찾지 못해 CPU로 자동 전환합니다.")
            cpu_variant = WHISPER_VARIANTS["cpu_base"]
            state["pipeline"] = _init_pipeline(cpu_variant)
            state["variant"] = cpu_variant
        else:
            raise
    print("Whisper 로딩 완료!\n")

    _EXCLUDE_DIRS = {"GT"}

    # 피싱 데이터 처리
    if os.path.exists(PHISHING_DIR):
        categories = sorted([d for d in os.listdir(PHISHING_DIR)
                            if os.path.isdir(os.path.join(PHISHING_DIR, d))
                            and d not in _EXCLUDE_DIRS])
        for cat in categories:
            cat_dir = os.path.join(PHISHING_DIR, cat)
            files = collect_audio_files(cat_dir)
            print(f"\n[피싱/{cat}] {len(files)}개 파일")
            completed_set = completed_phishing[cat]
            transcribe_category(state, enhancer, files, "phishing", cat,
                              phishing_csv, completed_set,
                              args.batch_size,
                              show_progress=not args.no_progress)

    # 일반 데이터 처리
    if os.path.exists(NORMAL_DIR):
        categories = sorted([d for d in os.listdir(NORMAL_DIR)
                            if os.path.isdir(os.path.join(NORMAL_DIR, d))
                            and d not in _EXCLUDE_DIRS])
        for cat in categories:
            cat_dir = os.path.join(NORMAL_DIR, cat)
            files = collect_audio_files(cat_dir)
            print(f"\n[일반/{cat}] {len(files)}개 파일")
            completed_set = completed_normal[cat]
            transcribe_category(state, enhancer, files, "normal", cat,
                              normal_csv, completed_set,
                              args.batch_size,
                              show_progress=not args.no_progress)

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
