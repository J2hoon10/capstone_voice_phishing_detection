# stt_preview.py
# STT 결과 미리보기: 피싱/일반 각 N개 샘플을 직접 전사하여 CSV로 저장
#
# 사용법:
#   python stt_preview.py                     # 피싱 10개 + 일반 10개 (기본)
#   python stt_preview.py --count 5           # 각 5개
#   python stt_preview.py --variant cpu_base  # CPU 모드

import os
import sys
import csv
import random
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_config import (
    PHISHING_DIR, NORMAL_DIR,
    AUDIO_EXTENSIONS, WHISPER_VARIANTS, DEFAULT_VARIANT,
    ERROR_ANALYSIS_DIR,
)
from audio_enhancer import AudioEnhancer
from faster_whisper import WhisperModel, BatchedInferencePipeline


OUTPUT_COLUMNS = ["label", "category", "filename", "stt_text", "enhance_sec", "stt_sec"]
MANIFEST_COLUMNS = ["label", "category", "filepath"]


def collect_per_category(base_dir, n_total, seed):
    """카테고리별로 균등하게 n_total개 샘플 수집"""
    EXCLUDE_DIRS = {"GT"}
    categories = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d not in EXCLUDE_DIRS
    ])

    all_files = {}
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        files = []
        for root, _, filenames in os.walk(cat_dir):
            for f in sorted(filenames):
                if f.lower().endswith(AUDIO_EXTENSIONS):
                    files.append((cat, os.path.join(root, f)))
        all_files[cat] = files

    rng = random.Random(seed)
    n_cats = len(categories)
    per_cat = max(1, n_total // n_cats)
    remainder = n_total - per_cat * n_cats

    sampled = []
    for i, cat in enumerate(categories):
        take = per_cat + (1 if i < remainder else 0)
        pool = all_files[cat]
        picked = rng.sample(pool, min(take, len(pool)))
        sampled.extend(picked)
        if len(pool) < take:
            print(f"  [WARN] '{cat}' 파일 부족: {len(pool)}개만 사용")

    return sampled


def load_manifest(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip()
            category = row.get("category", "").strip()
            filepath = row.get("filepath", "").strip()
            if label and category and filepath:
                rows.append((label, category, filepath))
    return rows


def save_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for label, category, filepath in rows:
            writer.writerow({
                "label": label,
                "category": category,
                "filepath": filepath,
            })


def init_pipeline(variant):
    model = WhisperModel(
        variant["model"],
        device=variant["device"],
        compute_type=variant["compute_type"],
    )
    return BatchedInferencePipeline(model=model)


def is_cuda_missing(err):
    msg = str(err)
    return ("cublas64_12.dll" in msg) or ("CUDA" in msg and "not found" in msg)


def transcribe_one(pipeline, audio_input, batch_size):
    segs, _ = pipeline.transcribe(
        audio_input,
        language="ko",
        batch_size=batch_size,
        beam_size=1,
        vad_filter=True,
        no_speech_threshold=0.85,
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        compression_ratio_threshold=2.2,
        log_prob_threshold=-0.8,
        temperature=(0.0, 0.2, 0.4, 0.6),
    )
    return " ".join([seg.text for seg in segs])


def main():
    parser = argparse.ArgumentParser(description="STT 결과 미리보기")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=list(WHISPER_VARIANTS.keys()))
    parser.add_argument("--count", type=int, default=10,
                        help="라벨별 샘플 수 (피싱 N개 + 일반 N개, 기본: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="출력 CSV 경로 (기본: error_analysis/stt_preview.csv)")
    parser.add_argument("--manifest", default=None,
                        help="고정 샘플 목록 CSV (기본: error_analysis/stt_preview_manifest.csv)")
    parser.add_argument("--regen-manifest", action="store_true",
                        help="기존 manifest 무시하고 새로 샘플 생성")
    args = parser.parse_args()

    variant = WHISPER_VARIANTS[args.variant]
    output_csv = args.output or os.path.join(ERROR_ANALYSIS_DIR, "stt_preview.csv")
    manifest_csv = args.manifest or os.path.join(ERROR_ANALYSIS_DIR, "stt_preview_manifest.csv")
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)

    print(f"=== STT 미리보기 ===")
    print(f"모델: {variant['model']} ({variant['device']}, {variant['compute_type']})")
    print(f"샘플: 피싱 {args.count}개 + 일반 {args.count}개")
    print()

    # 샘플 고정: manifest가 있으면 재사용, 없으면 생성 후 저장
    samples = None
    if not args.regen_manifest:
        samples = load_manifest(manifest_csv)

    if samples is None:
        phishing_samples = collect_per_category(PHISHING_DIR, args.count, args.seed)
        normal_samples = collect_per_category(NORMAL_DIR, args.count, args.seed)
        samples = [("phishing", cat, fp) for cat, fp in phishing_samples] + \
                  [("normal",   cat, fp) for cat, fp in normal_samples]
        save_manifest(manifest_csv, samples)
        print(f"[INFO] 고정 샘플 목록 저장: {manifest_csv}")
    else:
        print(f"[INFO] 고정 샘플 목록 사용: {manifest_csv}")

    p_cnt = sum(1 for s in samples if s[0] == "phishing")
    n_cnt = sum(1 for s in samples if s[0] == "normal")
    print(f"피싱 {p_cnt}개, 일반 {n_cnt}개 선택됨")
    print()

    # 모델 로드
    print("AudioEnhancer 로딩 중...")
    enhancer = AudioEnhancer(enable_bandpass=True, enable_denoise=True,
                             enable_vad=False, enable_normalize=True)
    print("Whisper 로딩 중...")
    state = {"variant": variant, "pipeline": None}
    try:
        state["pipeline"] = init_pipeline(variant)
    except RuntimeError as e:
        if variant["device"] == "cuda" and is_cuda_missing(e):
            print("[WARN] CUDA 없음 → CPU로 자동 전환")
            cpu_v = WHISPER_VARIANTS["cpu_base"]
            state["pipeline"] = init_pipeline(cpu_v)
            state["variant"] = cpu_v
        else:
            raise
    print("로딩 완료!\n")

    rows = []
    for idx, (label, category, filepath) in enumerate(samples, 1):
        fname = os.path.basename(filepath)
        print(f"[{idx:02d}/{len(samples)}] {label}/{category} — {fname}")

        # Enhance
        t0 = time.time()
        audio = enhancer.enhance(filepath)
        enhance_sec = round(time.time() - t0, 2)
        audio_input = audio if audio is not None else filepath

        # STT
        t1 = time.time()
        try:
            text = transcribe_one(state["pipeline"], audio_input,
                                  batch_size=8)
        except RuntimeError as e:
            if state["variant"]["device"] == "cuda" and is_cuda_missing(e):
                print("  [WARN] CUDA 없음 → CPU로 자동 전환")
                cpu_v = WHISPER_VARIANTS["cpu_base"]
                state["pipeline"] = init_pipeline(cpu_v)
                state["variant"] = cpu_v
                text = transcribe_one(state["pipeline"], audio_input, batch_size=8)
            else:
                raise
        stt_sec = round(time.time() - t1, 2)

        print(f"  → {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"  (enhance: {enhance_sec}s / stt: {stt_sec}s)")

        rows.append({
            "label":       label,
            "category":    category,
            "filename":    fname,
            "stt_text":    text,
            "enhance_sec": enhance_sec,
            "stt_sec":     stt_sec,
        })

    # CSV 저장
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== 완료 ===")
    print(f"결과 저장: {output_csv}")
    print(f"총 {len(rows)}건 (피싱 {sum(1 for r in rows if r['label']=='phishing')}개 / "
          f"일반 {sum(1 for r in rows if r['label']=='normal')}개)")


if __name__ == "__main__":
    main()
