# build_gt_from_conversations.py
# conversations.json의 스크립트+음성 쌍으로 ground_truth_annotated.csv 생성
# n/ 표기 및 짧은 발화 제거 → Whisper STT → whisper_error_analysis.py 입력 형식으로 저장
#
# 사용법:
#   python build_gt_from_conversations.py
#   python build_gt_from_conversations.py --variant cpu_base
#   python build_gt_from_conversations.py --resume

import os
import sys
import json
import csv
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_config import (
    CLASSIFIER_DIR, ERROR_ANALYSIS_DIR,
    WHISPER_VARIANTS, DEFAULT_VARIANT, BATCH_SIZE,
)
from audio_enhancer import AudioEnhancer
from faster_whisper import WhisperModel, BatchedInferencePipeline

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

CONVERSATIONS_JSON = os.path.join(CLASSIFIER_DIR, "conversations.json")
OUTPUT_CSV = os.path.join(ERROR_ANALYSIS_DIR, "ground_truth_annotated.csv")

OUTPUT_FIELDS = [
    "id", "filename", "label", "category",
    "stt_text", "human_transcription",
]

N_NOTATION_PREFIX = "n/"


def _is_cuda_missing_error(err):
    msg = str(err)
    return ("cublas64_12.dll" in msg) or ("CUDA" in msg and "not found" in msg)


def _init_pipeline(variant):
    model = WhisperModel(
        variant["model"],
        device=variant["device"],
        compute_type=variant["compute_type"],
    )
    return BatchedInferencePipeline(model=model)


def load_conversations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_valid_turns(conversations, wav_base, min_len):
    """n/ 제거, 짧은 텍스트 제거, wav 파일 존재 확인 후 유효 발화 목록 반환"""
    turns = []
    for session in conversations:
        session_id = session.get("session_id", "")
        category = session.get("metadata", {}).get("category", "교육")
        for turn in session.get("dialogs", []):
            text = (turn.get("text") or "").strip()
            wav_rel = (turn.get("wav_path") or "").strip()

            if not text or text.startswith(N_NOTATION_PREFIX):
                continue
            if len(text) < min_len:
                continue
            if not wav_rel:
                continue

            wav_abs = os.path.join(wav_base, wav_rel)
            if not os.path.exists(wav_abs):
                continue

            turns.append({
                "session_id": session_id,
                "turn": turn.get("turn", 0),
                "speaker_id": turn.get("speaker_id", ""),
                "category": category,
                "human_transcription": text,
                "wav_path": wav_abs,
                "wav_rel": wav_rel,
            })
    return turns


def load_completed(csv_path):
    """resume용: 이미 처리된 id 집합 반환"""
    done = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done.add(row["id"])
    return done


def transcribe_turns(state, enhancer, turns, output_csv, completed,
                     batch_size, show_progress):
    pending = [t for t in turns if _make_id(t) not in completed]
    skipped = len(turns) - len(pending)
    if skipped:
        print(f"  (이미 처리된 {skipped}건 건너뜀)")
    if not pending:
        return

    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    mode = "a" if file_exists else "w"

    iterator = pending
    if tqdm and show_progress:
        iterator = tqdm(pending, total=len(pending), desc="build-gt", unit="turn")

    with open(output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if not file_exists:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=1) as executor:
            next_future = executor.submit(enhancer.enhance, pending[0]["wav_path"])

            for i, turn in enumerate(iterator, 1):
                if tqdm and show_progress and hasattr(iterator, "set_postfix_str"):
                    iterator.set_postfix_str(os.path.basename(turn["wav_path"]))

                current_future = next_future
                if i < len(pending):
                    next_future = executor.submit(enhancer.enhance, pending[i]["wav_path"])
                else:
                    next_future = None

                enhanced = current_future.result()
                if enhanced is None:
                    audio_input = turn["wav_path"]
                elif isinstance(enhanced, np.ndarray):
                    audio_input = enhanced
                else:
                    audio_input = turn["wav_path"]

                def _transcribe(pipeline):
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
                    return " ".join(seg.text for seg in segs).strip()

                try:
                    stt_text = _transcribe(state["pipeline"])
                except RuntimeError as e:
                    if state["variant"]["device"] == "cuda" and _is_cuda_missing_error(e):
                        print("[WARN] CUDA 라이브러리 없음 → CPU 전환")
                        cpu_v = WHISPER_VARIANTS["cpu_base"]
                        state["pipeline"] = _init_pipeline(cpu_v)
                        state["variant"] = cpu_v
                        stt_text = _transcribe(state["pipeline"])
                    else:
                        raise

                uid = _make_id(turn)
                writer.writerow({
                    "id": uid,
                    "filename": os.path.basename(turn["wav_path"]),
                    "label": 0,
                    "category": turn["category"],
                    "stt_text": stt_text,
                    "human_transcription": turn["human_transcription"],
                })
                f.flush()
                completed.add(uid)


def _make_id(turn):
    return f"{turn['session_id']}_t{turn['turn']:04d}"


def main():
    parser = argparse.ArgumentParser(description="conversations.json → ground_truth_annotated.csv")
    parser.add_argument("--json", default=CONVERSATIONS_JSON, help="conversations.json 경로")
    parser.add_argument("--wav-base", default=CLASSIFIER_DIR, help="wav_path 기준 디렉터리")
    parser.add_argument("--output", default=OUTPUT_CSV, help="출력 CSV 경로")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        choices=list(WHISPER_VARIANTS.keys()), help="Whisper 변형")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--min-len", type=int, default=10, help="최소 텍스트 길이 (문자)")
    parser.add_argument("--resume", action="store_true", help="기존 결과 이어서 처리")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    conversations = load_conversations(args.json)
    turns = collect_valid_turns(conversations, args.wav_base, args.min_len)
    print(f"유효 발화: {len(turns)}건")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    completed = load_completed(args.output) if args.resume else set()
    if args.resume and completed:
        print(f"Resume: {len(completed)}건 완료됨")
    elif not args.resume and os.path.exists(args.output):
        os.remove(args.output)

    variant = WHISPER_VARIANTS[args.variant]
    print(f"Whisper: {variant['model']} ({variant['device']}, {variant['compute_type']})")

    print("AudioEnhancer 로딩 중...")
    enhancer = AudioEnhancer(enable_vad=False)

    print("Whisper 로딩 중...")
    state = {"variant": variant, "pipeline": None}
    try:
        state["pipeline"] = _init_pipeline(variant)
    except RuntimeError as e:
        if variant["device"] == "cuda" and _is_cuda_missing_error(e):
            print("[WARN] CUDA 없음 → CPU 전환")
            cpu_v = WHISPER_VARIANTS["cpu_base"]
            state["pipeline"] = _init_pipeline(cpu_v)
            state["variant"] = cpu_v
        else:
            raise
    print("준비 완료\n")

    transcribe_turns(
        state, enhancer, turns, args.output, completed,
        args.batch_size, show_progress=not args.no_progress,
    )

    print(f"\n완료: {args.output}")
    print("다음 단계: python whisper_error_analysis.py")


if __name__ == "__main__":
    main()
