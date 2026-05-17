"""
run_pipeline_watch.py

파일 기반 비동기 파이프라인:
  - STT  스레드: 30초 청크로 오디오 처리 → transcript.jsonl 에 문장 append
  - 분류기 스레드: transcript.jsonl 을 폴링 → 새 문장 감지 시 incremental 추론

파일 포맷 (transcript.jsonl):
  한 줄 = {"chunk": int, "audio_time": float, "text": "..."}

사용법:
  # 30초 청크 + 10초 오버랩 (기본)
  python run_pipeline_watch.py --audio /home/j2hoon10/20.mp3

  # 청크/오버랩 직접 지정
  python run_pipeline_watch.py --audio /home/j2hoon10/20.mp3 --chunk-secs 30 --overlap-secs 10

  # 분류기 폴링 간격 조절 (기본 0.5초)
  python run_pipeline_watch.py --audio /home/j2hoon10/20.mp3 --poll-interval 1.0

  # transcript 파일 저장 위치 지정
  python run_pipeline_watch.py --audio /home/j2hoon10/20.mp3 --transcript /tmp/transcript.jsonl
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── 분류기 실험 경로 ──────────────────────────────────────────────────────────
EXP_DIR = (
    Path(__file__).resolve().parents[1]
    / "models" / "classifier" / "experiments" / "roberta_mamba_freeze_init_4class"
)
sys.path.insert(0, str(EXP_DIR))

from config import CHECKPOINT_DIR, DEVICE, IDX_TO_LABEL, LABELS, ENCODER_CONFIG
from dataset import build_segments
from model import build_model
from transformers import AutoTokenizer

# ── 상수 ─────────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "상담 대화":       "\033[94m",
    "일상 대화":       "\033[92m",
    "대출 사기형":     "\033[93m",
    "수사기관 사칭형":  "\033[91m",
}
RESET   = "\033[0m"
BOLD    = "\033[1m"
EXP_NAME  = "roberta_mamba_freeze_init_4class"
WHISPER_SR = 16_000


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def load_checkpoint(run_id: str | None = None) -> Path:
    if run_id:
        return CHECKPOINT_DIR / f"{EXP_NAME}_{run_id}_best.pt"
    meta = CHECKPOINT_DIR / f"{EXP_NAME}_latest.json"
    if meta.exists():
        return Path(json.loads(meta.read_text(encoding="utf-8"))["checkpoint_path"])
    raise FileNotFoundError(f"체크포인트 없음: {CHECKPOINT_DIR}")


def get_temperature(k: int, warmup: int = 8, max_temp: float = 4.0) -> float:
    if k >= warmup:
        return 1.0
    return max_temp - (max_temp - 1.0) * (k - 1) / (warmup - 1)


def bar(prob: float, width: int = 8) -> str:
    return "█" * int(prob * width) + "░" * (width - int(prob * width))


# ── RoBERTa 단일 세그먼트 인코딩 ─────────────────────────────────────────────
@torch.no_grad()
def encode_window(model, seg: dict) -> torch.Tensor:
    ids  = torch.tensor([seg["input_ids"]],      dtype=torch.long, device=DEVICE)
    attn = torch.tensor([seg["attention_mask"]], dtype=torch.long, device=DEVICE)
    enc_dtype = next(model.encoder.parameters()).dtype
    H    = model.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
    return model.pooling(H, attn).to(enc_dtype).float()   # (1, D)


# ── Mamba 출력 → 4-class 확률 ────────────────────────────────────────────────
def predict_from_mamba_out(model, mamba_out: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    head       = model.head(mamba_out)
    super_p    = torch.softmax(head["super_logits"],    dim=-1)
    normal_p   = torch.softmax(head["normal_logits"],   dim=-1)
    phishing_p = torch.softmax(head["phishing_logits"], dim=-1)
    logits_4   = torch.cat([super_p[:, 0:1] * normal_p,
                             super_p[:, 1:2] * phishing_p], dim=-1)
    return F.softmax(torch.log(logits_4 + 1e-8) / temp, dim=-1)[0].cpu()


# ── 분류기 상태 ───────────────────────────────────────────────────────────────
def make_classifier_state():
    return {
        "accumulated":         [],
        "encoded_cache":       [],
        "processed_seg_count": 0,
        "last_probs":          None,
    }


@torch.no_grad()
def classifier_step(state: dict, new_text: str, classifier, tokenizer,
                    temp_warmup: int, temp_max: float) -> tuple[str | None, float, int]:
    state["accumulated"].append(new_text)
    full_text = " ".join(state["accumulated"])
    segments  = build_segments(tokenizer, full_text)

    new_segs = segments[state["processed_seg_count"]:]
    if not new_segs:
        return None, 0.0, 0

    for seg in new_segs:
        state["encoded_cache"].append(encode_window(classifier, seg))
        state["processed_seg_count"] += 1

    x = torch.stack(state["encoded_cache"], dim=1)
    y = x
    for mamba, drop in zip(classifier.mamba_layers, classifier.mamba_dropouts):
        y = drop(mamba(y))

    new_start = state["processed_seg_count"] - len(new_segs)
    for i in range(len(new_segs)):
        temp = get_temperature(new_start + i + 1, temp_warmup, temp_max)
        state["last_probs"] = predict_from_mamba_out(classifier, y[:, new_start + i, :], temp)

    probs = state["last_probs"]
    idx   = probs.argmax().item()
    return IDX_TO_LABEL[idx], probs[idx].item(), len(new_segs)


# ══════════════════════════════════════════════════════════════════════════════
# STT 스레드: 30초 청크 처리 → transcript.jsonl append
# ══════════════════════════════════════════════════════════════════════════════
def stt_worker(audio_path: Path, transcript_path: Path,
               whisper_model, chunk_secs: int, overlap_secs: int,
               done_event: threading.Event, stats: dict):
    """
    오디오를 chunk_secs 단위로 잘라 Whisper 처리.
    각 Whisper 세그먼트를 transcript_path 에 JSON 라인으로 append.
    완료 시 done_event.set().
    """
    import librosa

    print(f"\n[STT] 오디오 로드 중: {audio_path.name}")
    audio_np, _ = librosa.load(str(audio_path), sr=WHISPER_SR, mono=True)
    audio_duration = len(audio_np) / WHISPER_SR
    stats["audio_duration"] = audio_duration

    print(f"[STT] 오디오 길이: {audio_duration:.1f}s | "
          f"청크: {chunk_secs}s | 오버랩: {overlap_secs}s")
    print(f"[STT] transcript → {transcript_path}")

    # transcript 파일 초기화 (기존 내용 삭제)
    transcript_path.write_text("", encoding="utf-8")

    total_stt_time = 0.0
    n_whisper_segs = 0
    chunk_idx = 0

    chunk_start = 0.0
    while chunk_start < audio_duration:
        chunk_end        = min(chunk_start + chunk_secs, audio_duration)
        overlap_start    = max(0.0, chunk_start - overlap_secs)
        audio_slice      = audio_np[int(overlap_start * WHISPER_SR):int(chunk_end * WHISPER_SR)]
        # 새 구간 기준 오프셋 (오버랩 부분은 이미 이전 청크에서 처리됨)
        new_portion_offset = chunk_start - overlap_start

        print(f"\n[STT] 청크 {chunk_idx}  [{chunk_start:.0f}s ~ {chunk_end:.0f}s]"
              f"  Whisper 입력: [{overlap_start:.0f}s ~ {chunk_end:.0f}s]")

        t0 = time.perf_counter()
        segs_raw, _ = whisper_model.transcribe(
            audio_slice, language="ko", vad_filter=True,
        )
        whisper_segs = list(segs_raw)
        stt_elapsed  = time.perf_counter() - t0
        total_stt_time += stt_elapsed

        # 새 구간에 속하는 세그먼트만 transcript 에 기록
        new_segs_w = [s for s in whisper_segs if s.start >= new_portion_offset - 0.05]
        print(f"[STT]   Whisper {stt_elapsed:.2f}s | "
              f"전체 세그먼트: {len(whisper_segs)}  신규: {len(new_segs_w)}")

        with open(transcript_path, "a", encoding="utf-8") as f:
            for seg in new_segs_w:
                text = seg.text.strip()
                if not text:
                    continue
                line = json.dumps({
                    "chunk":      chunk_idx,
                    "audio_time": chunk_start + seg.start - new_portion_offset,
                    "text":       text,
                }, ensure_ascii=False)
                f.write(line + "\n")
                f.flush()
                n_whisper_segs += 1

        chunk_start += chunk_secs
        chunk_idx   += 1

    stats["stt_time"]      = total_stt_time
    stats["n_whisper_segs"] = n_whisper_segs
    print(f"\n[STT] 완료  총 {n_whisper_segs}개 세그먼트  STT 시간: {total_stt_time:.2f}s")
    done_event.set()


# ══════════════════════════════════════════════════════════════════════════════
# 분류기 스레드: transcript.jsonl 폴링 → 새 줄 감지 시 추론
# ══════════════════════════════════════════════════════════════════════════════
def classifier_worker(transcript_path: Path,
                      classifier, tokenizer,
                      temp_warmup: int, temp_max: float,
                      poll_interval: float,
                      done_event: threading.Event,
                      stats: dict):
    """
    transcript_path 를 poll_interval 초마다 폴링.
    새 JSON 라인이 추가될 때마다 분류기 추론 실행.
    STT done_event + 파일 처리 완료 시 종료.
    """
    print(f"\n[분류기] 폴링 시작 (간격: {poll_interval}s)")
    print(f"  {'시각':>6}  {'청크':>4}  {'텍스트':<32}  {'윈도우':>4}  "
          f"{'예측':<14}  {'확신도':>6}  확률 분포")
    print(f"  {'-'*90}")

    state         = make_classifier_state()
    lines_read    = 0
    total_inf_time = 0.0
    n_windows     = 0

    while True:
        # 파일에서 새 줄 읽기
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
        except FileNotFoundError:
            time.sleep(poll_interval)
            continue

        new_lines = all_lines[lines_read:]

        for raw in new_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue

            text       = entry.get("text", "")
            chunk_idx  = entry.get("chunk", -1)
            audio_time = entry.get("audio_time", 0.0)

            if not text:
                continue

            prev_label = (IDX_TO_LABEL[state["last_probs"].argmax().item()]
                          if state["last_probs"] is not None else None)

            t0 = time.perf_counter()
            pred_label, conf, n_new = classifier_step(
                state, text, classifier, tokenizer, temp_warmup, temp_max
            )
            inf_elapsed    = time.perf_counter() - t0
            total_inf_time += inf_elapsed
            n_windows      += n_new

            if pred_label is None:
                lines_read += 1
                continue

            changed = "◀" if (prev_label and pred_label != prev_label) else ""
            color   = LABEL_COLORS.get(pred_label, "")
            short   = text[:30] + "…" if len(text) > 30 else text.ljust(32)

            print(f"  {audio_time:>5.1f}s  {chunk_idx:>4}  {short:<32}  "
                  f"{state['processed_seg_count']:>4}  "
                  f"{color}{pred_label:<14}{RESET}  {conf:>5.1%}  ", end="")
            for lbl in LABELS:
                p = state["last_probs"][LABELS.index(lbl)].item()
                print(f"{lbl[:2]}:{bar(p)}{p:.2f}  ", end="")
            if changed:
                print(f" {BOLD}{changed}{RESET}", end="")
            print()

            lines_read += 1

        # STT가 끝났고 새 줄도 없으면 종료
        if done_event.is_set() and lines_read >= len(all_lines):
            break

        time.sleep(poll_interval)

    stats["inf_time"] = total_inf_time
    stats["n_windows"] = n_windows

    final = (IDX_TO_LABEL[state["last_probs"].argmax().item()]
             if state["last_probs"] is not None else "판단 불가")
    stats["final_prediction"] = final

    print(f"\n{'='*70}")
    print(f"  최종 판단: {BOLD}{LABEL_COLORS.get(final,'')}{final}{RESET}")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# 통합 실행
# ══════════════════════════════════════════════════════════════════════════════
def run_watch(audio_path: Path, transcript_path: Path,
              whisper_model, classifier, tokenizer,
              chunk_secs: int, overlap_secs: int,
              poll_interval: float,
              temp_warmup: int, temp_max: float):

    print(f"\n{'='*70}")
    print(f"  [파일 감시 파이프라인] {audio_path.name}")
    print(f"  STT  → {transcript_path.name}  (30초 청크 업데이트)")
    print(f"  분류기 ← {transcript_path.name}  (폴링 간격: {poll_interval}s)")
    print(f"{'='*70}")

    stats      = {}
    done_event = threading.Event()

    stt_thread = threading.Thread(
        target=stt_worker,
        args=(audio_path, transcript_path, whisper_model,
              chunk_secs, overlap_secs, done_event, stats),
        daemon=True,
    )
    clf_thread = threading.Thread(
        target=classifier_worker,
        args=(transcript_path, classifier, tokenizer,
              temp_warmup, temp_max, poll_interval, done_event, stats),
        daemon=True,
    )

    t_wall_start = time.perf_counter()
    stt_thread.start()
    clf_thread.start()

    stt_thread.join()
    clf_thread.join()
    t_wall_total = time.perf_counter() - t_wall_start

    # ── 속도 측정 출력 ─────────────────────────────────────────────────────
    audio_dur   = stats.get("audio_duration", 0.0)
    stt_time    = stats.get("stt_time", 0.0)
    inf_time    = stats.get("inf_time", 0.0)
    n_whisper   = stats.get("n_whisper_segs", 0)
    n_windows   = stats.get("n_windows", 0)
    total_time  = stt_time + inf_time
    rtf_seq     = total_time / audio_dur   if audio_dur   > 0 else float("inf")
    rtf_wall    = t_wall_total / audio_dur if audio_dur   > 0 else float("inf")
    stt_fps     = n_whisper  / stt_time    if stt_time    > 0 else float("inf")
    inf_fps     = n_windows  / inf_time    if inf_time    > 0 else float("inf")
    pip_fps     = n_windows  / total_time  if total_time  > 0 else float("inf")

    print(f"\n  [속도 측정 결과]")
    print(f"  {'항목':<38} {'값':>12}")
    print(f"  {'-'*52}")
    print(f"  {'오디오 길이':<38} {audio_dur:>11.2f}s")
    print(f"  {'STT 시간 (순수)':<38} {stt_time:>11.3f}s  ({stt_time/total_time*100:.1f}%)")
    print(f"  {'추론 시간 (순수)':<38} {inf_time:>11.3f}s  ({inf_time/total_time*100:.1f}%)")
    print(f"  {'합계 처리 시간 (순차)':<38} {total_time:>11.3f}s")
    print(f"  {'실제 벽시계 시간 (병렬)':<38} {t_wall_total:>11.3f}s")
    print(f"  {'-'*52}")
    rtf_flag = lambda r: "✓ 실시간 이하" if r < 1.0 else "✗ 실시간 초과"
    print(f"  {'RTF (순차 기준)':<38} {rtf_seq:>11.3f}x  {rtf_flag(rtf_seq)}")
    print(f"  {'RTF (벽시계 기준, 병렬 포함)':<38} {rtf_wall:>11.3f}x  {rtf_flag(rtf_wall)}")
    print(f"  {'-'*52}")
    print(f"  {'STT FPS (Whisper seg/s)':<38} {stt_fps:>11.2f}")
    print(f"  {'추론 FPS (classifier window/s)':<38} {inf_fps:>11.2f}")
    print(f"  {'파이프라인 FPS (window/s, 순차)':<38} {pip_fps:>11.2f}")
    print(f"  {'-'*52}")
    print(f"  {'Whisper 세그먼트 수':<38} {n_whisper:>12}")
    print(f"  {'분류기 윈도우 수':<38} {n_windows:>12}")

    return {
        "file": str(audio_path), "audio_duration": audio_dur,
        "stt_time": stt_time, "inf_time": inf_time,
        "total_time": total_time, "wall_time": t_wall_total,
        "rtf_sequential": rtf_seq, "rtf_wall": rtf_wall,
        "stt_fps": stt_fps, "inf_fps": inf_fps, "pipeline_fps": pip_fps,
        "n_whisper_segments": n_whisper, "n_classifier_windows": n_windows,
        "final_prediction": stats.get("final_prediction", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="파일 감시 STT + 분류기 파이프라인")
    parser.add_argument("--audio",         required=True,  help="오디오 파일 경로")
    parser.add_argument("--transcript",    default=None,   help="transcript.jsonl 저장 경로 (기본: audio와 동일 디렉터리)")
    parser.add_argument("--whisper-model", default="small", help="Whisper 모델 크기 (tiny/base/small/medium)")
    parser.add_argument("--run-id",        default=None,   help="분류기 체크포인트 run_id")
    parser.add_argument("--chunk-secs",    type=int,   default=30,  help="STT 청크 길이 (초, 기본: 30)")
    parser.add_argument("--overlap-secs",  type=int,   default=10,  help="Whisper 컨텍스트 오버랩 (초, 기본: 10)")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="분류기 폴링 간격 (초, 기본: 0.5)")
    parser.add_argument("--temp-warmup",   type=int,   default=8,   help="온도 스케일링 warmup 윈도우 수")
    parser.add_argument("--temp-max",      type=float, default=4.0, help="초반 최대 temperature")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[오류] 파일 없음: {audio_path}")
        sys.exit(1)

    if args.transcript:
        transcript_path = Path(args.transcript)
    else:
        transcript_path = audio_path.parent / f"{audio_path.stem}_transcript.jsonl"

    # ── Whisper 로드 ────────────────────────────────────────────────────────
    from faster_whisper import WhisperModel
    print(f"[로드] Whisper {args.whisper_model} ...")
    import torch as _torch
    _whisper_device = "cuda" if _torch.cuda.is_available() else "cpu"
    _compute_type   = "float16" if _whisper_device == "cuda" else "int8"
    print(f"[로드] Whisper device={_whisper_device}  compute_type={_compute_type}")
    whisper_model = WhisperModel(args.whisper_model, device=_whisper_device, compute_type=_compute_type)

    # ── 분류기 로드 ─────────────────────────────────────────────────────────
    ckpt_path = load_checkpoint(args.run_id)
    print(f"[로드] 분류기 체크포인트: {ckpt_path}")
    classifier = build_model().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(
        ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    )
    classifier.eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])

    run_watch(
        audio_path=audio_path,
        transcript_path=transcript_path,
        whisper_model=whisper_model,
        classifier=classifier,
        tokenizer=tokenizer,
        chunk_secs=args.chunk_secs,
        overlap_secs=args.overlap_secs,
        poll_interval=args.poll_interval,
        temp_warmup=args.temp_warmup,
        temp_max=args.temp_max,
    )


if __name__ == "__main__":
    main()
