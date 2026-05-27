"""
compare_streaming.py

여러 실험 모델의 streaming inference 결과를 나란히 비교 출력.

사용법:
  python compare_streaming.py --text "대화 내용..."
  python compare_streaming.py --text "..." --models mamba gru lstm
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 지원 모델 목록 ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "mamba":    "roberta_mamba_freeze_init_4class",
    "mamba_w32":"roberta_mamba_w32_freeze_init_4class",
    "gru":      "roberta_gru_freeze_init_4class",
    "lstm":     "roberta_lstm_freeze_init_4class",
    "avgpool":  "roberta_avgpool_4class",
}

LABEL_SHORT = {
    "상담 대화":      "상담",
    "일상 대화":      "일상",
    "대출 사기형":    "대출",
    "수사기관 사칭형": "수사기관",
}

LABEL_COLORS = {
    "상담 대화":      "\033[94m",
    "일상 대화":      "\033[92m",
    "대출 사기형":    "\033[93m",
    "수사기관 사칭형": "\033[91m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"


def bar(prob: float, width: int = 6) -> str:
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


def get_temperature(k: int, warmup: int = 8, max_temp: float = 4.0) -> float:
    if k >= warmup:
        return 1.0
    return max_temp - (max_temp - 1.0) * (k - 1) / (warmup - 1)


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model(exp_name: str):
    exp_dir = PROJECT_ROOT / "models" / "experiments" / "model_architecture" / exp_name
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))

    # 모듈 충돌 방지: 이미 로드된 모듈 제거
    for mod in ["config", "model", "dataset"]:
        sys.modules.pop(mod, None)

    from config import CHECKPOINT_DIR, DEVICE, IDX_TO_LABEL, LABELS, ENCODER_CONFIG, MAX_SEQ_LEN
    from model import build_model
    from dataset import build_segments
    from transformers import AutoTokenizer

    meta = CHECKPOINT_DIR / f"{exp_name}_latest.json"
    if not meta.exists():
        raise FileNotFoundError(f"latest.json 없음: {meta}")
    ckpt_path = Path(json.loads(meta.read_text(encoding="utf-8"))["checkpoint_path"])

    model = build_model().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])

    return {
        "model":         model,
        "tokenizer":     tokenizer,
        "build_segments": build_segments,
        "DEVICE":        DEVICE,
        "IDX_TO_LABEL":  IDX_TO_LABEL,
        "LABELS":        LABELS,
        "MAX_SEQ_LEN":   MAX_SEQ_LEN,
        "exp_name":      exp_name,
    }


# ── Incremental 추론 ───────────────────────────────────────────────────────────
@torch.no_grad()
def run_streaming(ctx: dict, text: str, temp_warmup: int = 8, temp_max: float = 4.0):
    model         = ctx["model"]
    tokenizer     = ctx["tokenizer"]
    build_segments= ctx["build_segments"]
    DEVICE        = ctx["DEVICE"]
    IDX_TO_LABEL  = ctx["IDX_TO_LABEL"]
    LABELS        = ctx["LABELS"]
    MAX_SEQ_LEN   = ctx["MAX_SEQ_LEN"]

    segments = build_segments(tokenizer, text)
    results  = []

    for k in range(1, len(segments) + 1):
        segs    = segments[:k]
        max_len = min(max(len(s["input_ids"]) for s in segs), MAX_SEQ_LEN)
        S       = k
        input_ids      = torch.zeros(1, S, max_len, dtype=torch.long)
        attention_mask = torch.zeros(1, S, max_len, dtype=torch.long)
        segment_mask   = torch.ones(1, S, dtype=torch.bool)
        for j, seg in enumerate(segs):
            ids  = seg["input_ids"][:max_len]
            attn = seg["attention_mask"][:max_len]
            L = len(ids)
            input_ids[0, j, :L]      = torch.tensor(ids,  dtype=torch.long)
            attention_mask[0, j, :L] = torch.tensor(attn, dtype=torch.long)

        batch = {
            "input_ids":      input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE),
            "segment_mask":   segment_mask.to(DEVICE),
            "num_segments":   torch.tensor([S], dtype=torch.long).to(DEVICE),
        }
        out      = model(**batch)
        temp     = get_temperature(k, temp_warmup, temp_max)
        probs    = F.softmax(out["logits"].float() / temp, dim=-1)[0].cpu()
        pred_idx = probs.argmax().item()
        results.append({
            "window":    k,
            "label":     IDX_TO_LABEL[pred_idx],
            "conf":      probs[pred_idx].item(),
            "probs":     {IDX_TO_LABEL[i]: probs[i].item() for i in range(len(LABELS))},
            "temp":      temp,
        })

    return results, len(segments)


# ── 비교 출력 ──────────────────────────────────────────────────────────────────
def print_comparison(model_names: list[str], all_results: list, n_windows: int,
                     true_label: str | None = None):
    n_models = len(model_names)
    col_w    = 22   # 모델당 열 너비

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  스트리밍 추론 비교  |  총 윈도우: {n_windows}개")
    if true_label:
        print(f"  정답 레이블: {BOLD}{true_label}{RESET}")
    print(f"{'='*75}")

    header = f"  {'윈도':>4}  "
    for name in model_names:
        header += f"{name.upper():<{col_w}}"
    print(header)
    print(f"  {'-'*70}")

    # ── 윈도우별 행 ───────────────────────────────────────────────────────────
    prev_labels = [None] * n_models
    for w in range(n_windows):
        row = f"  {w+1:>4}  "
        for i, res_list in enumerate(all_results):
            r     = res_list[w]
            label = r["label"]
            conf  = r["conf"]
            color = LABEL_COLORS.get(label, "")
            changed = "◀" if (prev_labels[i] and label != prev_labels[i]) else " "
            short = LABEL_SHORT.get(label, label)
            cell  = f"{color}{short:<6}{RESET} {conf:>5.1%} {changed}"
            row  += f"{cell:<{col_w}}"
            prev_labels[i] = label
        print(row)

    # ── 확률 분포 섹션 ────────────────────────────────────────────────────────
    print(f"\n  {'윈도':>4}  {'모델':<12}  {'상담':>10}  {'일상':>10}  {'대출':>10}  {'수사기관':>10}")
    print(f"  {'-'*70}")
    LABELS_ORDER = ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]
    for w in range(n_windows):
        for i, (name, res_list) in enumerate(zip(model_names, all_results)):
            r   = res_list[w]
            row = f"  {w+1:>4}  {name.upper():<12}"
            for lbl in LABELS_ORDER:
                p    = r["probs"][lbl]
                color = LABEL_COLORS.get(lbl, "")
                row += f"  {color}{bar(p)}{p:.2f}{RESET}"
            print(row)
        if w < n_windows - 1:
            print(f"  {'':>4}  {'─'*60}")

    # ── 최종 예측 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {'모델':<14}  {'최종 예측':<16}  {'확신도':>6}  {'정답 여부'}")
    print(f"  {'-'*55}")
    for name, res_list in zip(model_names, all_results):
        final = res_list[-1]
        label = final["label"]
        conf  = final["conf"]
        color = LABEL_COLORS.get(label, "")
        if true_label:
            correct = f"{GREEN}✓ 정답{RESET}" if label == true_label else f"{RED}✗ 오답{RESET}"
        else:
            correct = ""
        print(f"  {name.upper():<14}  {color}{label:<16}{RESET}  {conf:>5.1%}  {correct}")
    print(f"{'='*75}\n")


# ── 진입점 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="스트리밍 추론 모델 비교")
    parser.add_argument("--text",        required=True,  help="비교할 대화 텍스트")
    parser.add_argument("--models",      nargs="+",
                        default=["mamba", "gru", "lstm"],
                        choices=list(MODEL_REGISTRY.keys()),
                        help="비교할 모델 (기본: mamba gru lstm)")
    parser.add_argument("--true-label",  default=None,   help="정답 레이블 (선택)")
    parser.add_argument("--temp-warmup", type=int,   default=8)
    parser.add_argument("--temp-max",    type=float, default=4.0)
    args = parser.parse_args()

    all_results  = []
    model_names  = []
    n_windows    = 0

    for key in args.models:
        exp_name = MODEL_REGISTRY[key]
        print(f"[로드] {key.upper()} ({exp_name}) ...", flush=True)
        ctx = load_model(exp_name)
        results, n_win = run_streaming(ctx, args.text, args.temp_warmup, args.temp_max)
        all_results.append(results)
        model_names.append(key)
        n_windows = n_win
        print(f"       완료 (윈도우 {n_win}개)")

    print_comparison(model_names, all_results, n_windows, args.true_label)


if __name__ == "__main__":
    main()
