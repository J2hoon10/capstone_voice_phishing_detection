"""
run_pipeline.py

음성 파일 → Whisper STT → RoBERTa-Mamba 분류기 스트리밍 파이프라인
STT + 추론 속도를 FPS(프레임/초) 및 RTF(실시간 배율)로 측정.

분류기: --exp 인자로 선택 (기본: w64)
  - w64: roberta_mamba_freeze_init_4class        (WINDOW=64, STRIDE=32, MAX_SEQ=128)
  - w32: roberta_mamba_w32_freeze_init_4class    (WINDOW=32, STRIDE=16, MAX_SEQ=64)

모드:
  [기본] 파일 전체를 한 번에 Whisper 처리
  [청크] --chunk-secs N  →  N초 단위로 오디오를 잘라 순차 처리
                            --overlap-secs M  →  이전 M초를 Whisper 컨텍스트로 포함
                            ex) chunk=30, overlap=10 → [0:30], [20:60], [50:90], ...

사용법:
  # W64 기본 (30초 청크)
  python run_pipeline.py --audio /home/j2hoon10/20.mp3 --chunk-secs 30 --overlap-secs 10

  # W32 모델로 전환
  python run_pipeline.py --audio /home/j2hoon10/20.mp3 --chunk-secs 15 --overlap-secs 8 --exp w32

  # 디렉터리 전체
  python run_pipeline.py --audio /home/j2hoon10/ --ext mp3 --chunk-secs 30 --overlap-secs 10
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── 실험 이름 → 디렉터리 매핑 ──────────────────────────────────────────────────
EXPERIMENTS = {
    "w64": "roberta_mamba_freeze_init_4class",
    "w32": "roberta_mamba_w32_freeze_init_4class",
}

MODELS_BASE = Path(__file__).resolve().parents[1] / "models" / "experiments" / "model_architecture"


def _load_exp_modules(exp_key: str):
    """실험 디렉터리를 sys.path 앞에 삽입하고 config/dataset/model 모듈을 동적 로드."""
    exp_name = EXPERIMENTS[exp_key]
    exp_dir  = str(MODELS_BASE / exp_name)

    # 이전 실험 경로가 남아 있으면 제거 후 새 경로 삽입
    sys.path = [p for p in sys.path if str(MODELS_BASE) not in p]
    sys.path.insert(0, exp_dir)

    # 캐시된 모듈 제거 (다른 exp를 이미 로드했을 경우 대비)
    for mod in ("config", "dataset", "model"):
        sys.modules.pop(mod, None)

    config  = importlib.import_module("config")
    dataset = importlib.import_module("dataset")
    model   = importlib.import_module("model")
    return config, dataset, model


from transformers import AutoTokenizer

# ── 색상 코드 ─────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "상담 대화":       "\033[94m",
    "일상 대화":       "\033[92m",
    "대출 사기형":     "\033[93m",
    "수사기관 사칭형":  "\033[91m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"

WHISPER_SR = 16_000   # faster-whisper 입력 샘플레이트


# ── 체크포인트 로드 ───────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_dir: Path, exp_name: str, run_id: str | None = None) -> Path:
    if run_id:
        return checkpoint_dir / f"{exp_name}_{run_id}_best.pt"
    meta = checkpoint_dir / f"{exp_name}_latest.json"
    if meta.exists():
        return Path(json.loads(meta.read_text(encoding="utf-8"))["checkpoint_path"])
    raise FileNotFoundError(f"체크포인트 없음: {checkpoint_dir}")


# ── 온도 스케일링 ─────────────────────────────────────────────────────────────
def get_temperature(k: int, warmup: int = 8, max_temp: float = 4.0) -> float:
    if k >= warmup:
        return 1.0
    return max_temp - (max_temp - 1.0) * (k - 1) / (warmup - 1)


# ── RoBERTa 단일 세그먼트 인코딩 ─────────────────────────────────────────────
@torch.no_grad()
def encode_window(model, seg: dict, device) -> torch.Tensor:
    ids  = torch.tensor([seg["input_ids"]],      dtype=torch.long, device=device)
    attn = torch.tensor([seg["attention_mask"]], dtype=torch.long, device=device)
    enc_dtype = next(model.encoder.parameters()).dtype
    H   = model.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
    return model.pooling(H, attn).to(enc_dtype).float()   # (1, D)


# ── Mamba 출력 → 4-class 확률 ────────────────────────────────────────────────
def predict_from_mamba_out(model, mamba_out: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    head        = model.head(mamba_out)
    super_p     = torch.softmax(head["super_logits"],    dim=-1)
    normal_p    = torch.softmax(head["normal_logits"],   dim=-1)
    phishing_p  = torch.softmax(head["phishing_logits"], dim=-1)
    logits_4    = torch.cat([super_p[:, 0:1] * normal_p,
                              super_p[:, 1:2] * phishing_p], dim=-1)
    return F.softmax(torch.log(logits_4 + 1e-8) / temp, dim=-1)[0].cpu()


def bar(prob: float, width: int = 8) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


# ── 분류기 상태 초기화 ────────────────────────────────────────────────────────
def make_classifier_state():
    return {
        "accumulated":         [],    # 누적 문장 리스트
        "encoded_cache":       [],    # RoBERTa 인코딩 벡터 캐시 [(1,D), ...]
        "processed_seg_count": 0,     # 이미 처리한 윈도우 수
        "last_probs":          None,  # 마지막 4-class 확률
    }


# ── 분류기 incremental 추론 ───────────────────────────────────────────────────
@torch.no_grad()
def classifier_step(state: dict, new_text: str, classifier, tokenizer,
                    temp_warmup: int, temp_max: float,
                    build_segments_fn, idx_to_label: dict,
                    device) -> tuple[str | None, float, int]:
    """
    new_text 를 누적한 뒤 새로 생긴 윈도우만 인코딩 + Mamba 실행.
    반환: (pred_label, conf, n_new_windows)
    """
    state["accumulated"].append(new_text)
    full_text = " ".join(state["accumulated"])
    segments  = build_segments_fn(tokenizer, full_text)

    new_segs = segments[state["processed_seg_count"]:]
    if not new_segs:
        return None, 0.0, 0

    for seg in new_segs:
        state["encoded_cache"].append(encode_window(classifier, seg, device))
        state["processed_seg_count"] += 1

    x = torch.stack(state["encoded_cache"], dim=1)   # (1, T, D)
    y = x
    for mamba, drop in zip(classifier.mamba_layers, classifier.mamba_dropouts):
        y = drop(mamba(y))

    new_start = state["processed_seg_count"] - len(new_segs)
    for i in range(len(new_segs)):
        temp = get_temperature(new_start + i + 1, temp_warmup, temp_max)
        state["last_probs"] = predict_from_mamba_out(classifier, y[:, new_start + i, :], temp)

    probs = state["last_probs"]
    idx   = probs.argmax().item()
    return idx_to_label[idx], probs[idx].item(), len(new_segs)


# ── 타이밍 결과 출력 ─────────────────────────────────────────────────────────
def print_timing(audio_duration, stt_time, inf_time, n_whisper, n_windows) -> dict:
    total_time   = stt_time + inf_time
    rtf          = total_time / audio_duration if audio_duration > 0 else float("inf")
    stt_fps      = n_whisper / stt_time      if stt_time  > 0 else float("inf")
    inf_fps      = n_windows / inf_time      if inf_time  > 0 else float("inf")
    pipeline_fps = n_windows / total_time    if total_time > 0 else float("inf")

    print(f"\n  [속도 측정 결과]")
    print(f"  {'항목':<32} {'값':>12}")
    print(f"  {'-'*46}")
    print(f"  {'오디오 길이':<32} {audio_duration:>11.2f}s")
    print(f"  {'전체 처리 시간':<32} {total_time:>11.3f}s")
    print(f"  {'  STT (Whisper)':<32} {stt_time:>11.3f}s  ({stt_time/total_time*100:.1f}%)")
    print(f"  {'  추론 (분류기)':<32} {inf_time:>11.3f}s  ({inf_time/total_time*100:.1f}%)")
    print(f"  {'-'*46}")
    rtf_flag = "✓ 실시간 이하" if rtf < 1.0 else "✗ 실시간 초과"
    print(f"  {'RTF':<32} {rtf:>11.3f}x  {rtf_flag}")
    print(f"  {'-'*46}")
    print(f"  {'STT FPS (Whisper seg/s)':<32} {stt_fps:>11.2f}")
    print(f"  {'추론 FPS (classifier window/s)':<32} {inf_fps:>11.2f}")
    print(f"  {'파이프라인 FPS (window/s)':<32} {pipeline_fps:>11.2f}")
    print(f"  {'-'*46}")
    print(f"  {'Whisper 세그먼트 수':<32} {n_whisper:>12}")
    print(f"  {'분류기 윈도우 수':<32} {n_windows:>12}")

    return {
        "audio_duration": audio_duration,
        "stt_time": stt_time,
        "inf_time": inf_time,
        "total_time": total_time,
        "rtf": rtf,
        "stt_fps": stt_fps,
        "inf_fps": inf_fps,
        "pipeline_fps": pipeline_fps,
        "n_whisper_segments": n_whisper,
        "n_classifier_windows": n_windows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 모드 A: 파일 전체를 한 번에 처리
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_full(audio_path: Path, whisper_model, classifier, tokenizer,
             temp_warmup: int, temp_max: float,
             build_segments_fn, idx_to_label: dict, labels: list, device) -> dict | None:
    print(f"\n{'='*70}")
    print(f"  [전체 모드] {audio_path.name}")
    print(f"{'='*70}")

    state = make_classifier_state()
    t_stt_start = time.perf_counter()
    segs_raw, info = whisper_model.transcribe(str(audio_path), language="ko", vad_filter=True)
    whisper_segs = list(segs_raw)
    t_stt_end   = time.perf_counter()

    audio_duration = info.duration
    stt_time       = t_stt_end - t_stt_start

    if not whisper_segs:
        print("  [경고] Whisper 출력 없음")
        return None

    print(f"\n  오디오 길이: {audio_duration:.1f}s | STT: {stt_time:.2f}s "
          f"| Whisper 세그먼트: {len(whisper_segs)}")
    print(f"\n  {'Whisper 텍스트':<32} {'윈도우':>4}  {'예측':<14}  {'확신도':>6}  확률 분포")
    print(f"  {'-'*70}")

    t_inf_total = 0.0
    n_whisper   = 0

    for ws in whisper_segs:
        text = ws.text.strip()
        if not text:
            continue

        prev_label = idx_to_label[state["last_probs"].argmax().item()] if state["last_probs"] is not None else None
        t0 = time.perf_counter()
        pred_label, conf, n_new = classifier_step(
            state, text, classifier, tokenizer, temp_warmup, temp_max,
            build_segments_fn, idx_to_label, device,
        )
        t_inf_total += time.perf_counter() - t0
        n_whisper   += 1

        if pred_label is None:
            continue

        changed = "◀" if (prev_label and pred_label != prev_label) else ""
        color   = LABEL_COLORS.get(pred_label, "")
        short   = text[:30] + "…" if len(text) > 30 else text.ljust(32)
        print(f"  {short:<32} {state['processed_seg_count']:>4}  "
              f"{color}{pred_label:<14}{RESET}  {conf:>5.1%}  ", end="")
        for j, lbl in enumerate(labels):
            p = state["last_probs"][j].item()
            print(f"{lbl[:2]}:{bar(p)}{p:.2f}  ", end="")
        if changed:
            print(f" {BOLD}{changed}{RESET}", end="")
        print()

    final = idx_to_label[state["last_probs"].argmax().item()] if state["last_probs"] is not None else "판단 불가"
    print(f"\n{'='*70}")
    print(f"  최종 판단: {BOLD}{LABEL_COLORS.get(final,'')}{final}{RESET}")

    stats = print_timing(audio_duration, stt_time, t_inf_total, n_whisper, state["processed_seg_count"])
    stats["file"]             = str(audio_path)
    stats["final_prediction"] = final
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 모드 B: 청크 + 오버랩
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_chunked(audio_path: Path, whisper_model, classifier, tokenizer,
                chunk_secs: int, overlap_secs: int,
                temp_warmup: int, temp_max: float,
                build_segments_fn, idx_to_label: dict, device) -> dict | None:
    """
    오디오를 chunk_secs 단위로 잘라 순차 처리.
    각 청크는 이전 overlap_secs 초를 Whisper 컨텍스트로 포함.

    청크 구조 (chunk=30, overlap=10 예시):
      청크 0: Whisper 입력 [  0s, 30s]  → 전체 사용 (오버랩 없음)
      청크 1: Whisper 입력 [ 20s, 60s]  → 30s 이후 세그먼트만 분류기에 반영
      청크 2: Whisper 입력 [ 50s, 90s]  → 60s 이후 세그먼트만 반영
      ...
    """
    import librosa

    print(f"\n{'='*70}")
    print(f"  [청크 모드] {audio_path.name}")
    print(f"  chunk={chunk_secs}s  overlap={overlap_secs}s")
    print(f"{'='*70}")

    # 오디오 전체 로드 (16kHz mono)
    audio_np, _ = librosa.load(str(audio_path), sr=WHISPER_SR, mono=True)
    total_samples   = len(audio_np)
    audio_duration  = total_samples / WHISPER_SR
    chunk_samples   = int(chunk_secs   * WHISPER_SR)
    overlap_samples = int(overlap_secs * WHISPER_SR)

    n_chunks = max(1, int(np.ceil(total_samples / chunk_samples)))
    print(f"\n  오디오 길이: {audio_duration:.1f}s | 총 청크: {n_chunks}개\n")

    state = make_classifier_state()
    t_stt_total = 0.0
    t_inf_total = 0.0
    n_whisper   = 0

    print(f"  {'청크':>4}  {'구간':^16}  {'Whisper 텍스트':<28}  {'윈도우':>4}  {'예측':<14}  {'확신도':>6}")
    print(f"  {'-'*90}")

    for i in range(n_chunks):
        chunk_start_s   = i * chunk_secs
        chunk_end_s     = min((i + 1) * chunk_secs, audio_duration)
        overlap_start_s = max(0.0, chunk_start_s - overlap_secs)

        # 오디오 슬라이스 (overlap 포함)
        w_start = int(overlap_start_s * WHISPER_SR)
        w_end   = int(chunk_end_s     * WHISPER_SR)
        audio_slice = audio_np[w_start:w_end]

        # 슬라이스 내에서 "새 구간"의 시작 오프셋(초)
        new_portion_offset = chunk_start_s - overlap_start_s

        # ── STT ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        segs_raw, _ = whisper_model.transcribe(
            audio_slice,
            language="ko",
            vad_filter=True,
        )
        whisper_segs = list(segs_raw)
        t_stt_total += time.perf_counter() - t0

        # 새 구간(오버랩 이후)에 해당하는 세그먼트만 사용
        new_segs_w = [s for s in whisper_segs if s.start >= new_portion_offset]

        if not new_segs_w:
            print(f"  {i+1:>4}  [{chunk_start_s:5.1f}s~{chunk_end_s:5.1f}s]  "
                  f"{'(Whisper 출력 없음)':<28}")
            continue

        # ── 분류기 추론 ───────────────────────────────────────────────────────
        first_in_chunk = True
        for ws in new_segs_w:
            text = ws.text.strip()
            if not text:
                continue

            prev_label = (idx_to_label[state["last_probs"].argmax().item()]
                          if state["last_probs"] is not None else None)
            t0 = time.perf_counter()
            pred_label, conf, n_new = classifier_step(
                state, text, classifier, tokenizer, temp_warmup, temp_max,
                build_segments_fn, idx_to_label, device,
            )
            t_inf_total += time.perf_counter() - t0
            n_whisper   += 1

            if pred_label is None:
                continue

            changed = "◀" if (prev_label and pred_label != prev_label) else ""
            color   = LABEL_COLORS.get(pred_label, "")
            short   = text[:26] + "…" if len(text) > 26 else text.ljust(28)
            chunk_label = f"[{chunk_start_s:.0f}s~{chunk_end_s:.0f}s]" if first_in_chunk else " " * 16

            print(f"  {i+1:>4}  {chunk_label:<16}  {short:<28}  "
                  f"{state['processed_seg_count']:>4}  "
                  f"{color}{pred_label:<14}{RESET}  {conf:>5.1%}  ", end="")
            if changed:
                print(f"{BOLD}{changed}{RESET}", end="")
            print()
            first_in_chunk = False

    final = (idx_to_label[state["last_probs"].argmax().item()]
             if state["last_probs"] is not None else "판단 불가")
    print(f"\n{'='*70}")
    print(f"  최종 판단: {BOLD}{LABEL_COLORS.get(final,'')}{final}{RESET}")

    stats = print_timing(audio_duration, t_stt_total, t_inf_total, n_whisper, state["processed_seg_count"])
    stats["file"]             = str(audio_path)
    stats["final_prediction"] = final
    stats["n_chunks"]         = n_chunks
    return stats


# ── 오디오 파일 목록 수집 ─────────────────────────────────────────────────────
def collect_audio_files(path: Path, ext: str) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob(f"*.{ext}"))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Whisper STT + Mamba 분류 파이프라인 속도 측정")
    parser.add_argument("--audio",         required=True,  help="음성 파일 또는 디렉터리 경로")
    parser.add_argument("--ext",           default="mp3",  help="디렉터리 탐색 확장자 (기본: mp3)")
    parser.add_argument("--whisper-model", default="small", help="Whisper 모델 크기 (tiny/base/small/medium/large)")
    parser.add_argument("--run-id",        default=None,   help="분류기 체크포인트 run_id")
    parser.add_argument("--temp-warmup",   type=int,   default=8,   help="온도 warmup 윈도우 수")
    parser.add_argument("--temp-max",      type=float, default=4.0, help="초반 최대 temperature")
    # 청크 모드
    parser.add_argument("--chunk-secs",   type=int, default=None,
                        help="청크 모드: 청크 길이(초). 미지정 시 전체 파일 모드")
    parser.add_argument("--overlap-secs", type=int, default=10,
                        help="청크 모드: Whisper 컨텍스트용 오버랩 길이(초, 기본: 10)")
    parser.add_argument("--exp",          default="w64", choices=list(EXPERIMENTS.keys()),
                        help="분류기 실험 선택: w64 (기본, WINDOW=64) | w32 (WINDOW=32)")
    args = parser.parse_args()

    # ── 실험 모듈 동적 로드 ───────────────────────────────────────────────────
    exp_name = EXPERIMENTS[args.exp]
    print(f"[로드] 실험: {exp_name}  (--exp {args.exp})")
    cfg, ds, mdl = _load_exp_modules(args.exp)

    DEVICE_        = cfg.DEVICE
    IDX_TO_LABEL_  = cfg.IDX_TO_LABEL
    LABELS_        = cfg.LABELS
    ENCODER_CONFIG_= cfg.ENCODER_CONFIG
    CHECKPOINT_DIR_= cfg.CHECKPOINT_DIR
    build_segments_fn = ds.build_segments
    build_model_fn    = mdl.build_model

    # ── Whisper 로드 ──────────────────────────────────────────────────────────
    from faster_whisper import WhisperModel
    use_cuda     = torch.cuda.is_available()
    w_device     = "cuda" if use_cuda else "cpu"
    compute_type = "float16" if use_cuda else "int8"
    print(f"[로드] Whisper '{args.whisper_model}'  device={w_device}  compute={compute_type}")
    whisper_model = WhisperModel(args.whisper_model, device=w_device, compute_type=compute_type)

    # ── 분류기 로드 ───────────────────────────────────────────────────────────
    ckpt_path  = load_checkpoint(CHECKPOINT_DIR_, exp_name, args.run_id)
    print(f"[로드] 분류기: {ckpt_path}")
    classifier = build_model_fn().to(DEVICE_)
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(
        ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    )
    classifier.eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG_["MODEL_NAME"])

    # ── 음성 파일 수집 ────────────────────────────────────────────────────────
    audio_files = collect_audio_files(Path(args.audio), args.ext)
    if not audio_files:
        print(f"[오류] 음성 파일 없음: {args.audio}")
        sys.exit(1)
    print(f"[INFO] 처리 파일: {len(audio_files)}개\n")

    # ── 파일별 실행 ───────────────────────────────────────────────────────────
    all_results = []
    for af in audio_files:
        if args.chunk_secs:
            result = run_chunked(
                af, whisper_model, classifier, tokenizer,
                chunk_secs=args.chunk_secs, overlap_secs=args.overlap_secs,
                temp_warmup=args.temp_warmup, temp_max=args.temp_max,
                build_segments_fn=build_segments_fn,
                idx_to_label=IDX_TO_LABEL_, device=DEVICE_,
            )
        else:
            result = run_full(
                af, whisper_model, classifier, tokenizer,
                temp_warmup=args.temp_warmup, temp_max=args.temp_max,
                build_segments_fn=build_segments_fn,
                idx_to_label=IDX_TO_LABEL_, labels=LABELS_, device=DEVICE_,
            )
        if result:
            all_results.append(result)

    # ── 전체 요약 (파일 2개 이상) ─────────────────────────────────────────────
    if len(all_results) > 1:
        total_audio = sum(r["audio_duration"] for r in all_results)
        total_proc  = sum(r["total_time"]     for r in all_results)
        avg_rtf     = total_proc / total_audio
        avg_fps     = sum(r["pipeline_fps"] for r in all_results) / len(all_results)
        print(f"\n{'='*70}")
        print(f"  [전체 {len(all_results)}개 파일 요약]")
        print(f"  총 오디오: {total_audio:.1f}s  |  총 처리: {total_proc:.1f}s")
        print(f"  평균 RTF: {avg_rtf:.3f}x  |  평균 파이프라인 FPS: {avg_fps:.2f}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
