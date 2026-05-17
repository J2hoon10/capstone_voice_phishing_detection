"""
visualize_comparison.py

세 모델의 스트리밍 추론 결과를 가로 막대 그래프로 비교 시각화.
각 윈도우 × 각 모델별로 4-class 확률을 누적 가로 막대로 표현.

사용법:
  python visualize_comparison.py --text "대화 내용..." --true-label "상담 대화"
  python visualize_comparison.py --text "..." --output result.png
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import koreanize_matplotlib  # NanumGothic 한국어 폰트 자동 적용
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_REGISTRY = {
    "mamba":     "roberta_mamba_freeze_init_4class",
    "mamba_w32": "roberta_mamba_w32_freeze_init_4class",
    "gru":       "roberta_gru_freeze_init_4class",
    "lstm":      "roberta_lstm_freeze_init_4class",
    "avgpool":   "roberta_avgpool_4class",
}

LABELS_ORDER = ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]
CLASS_COLORS = {
    "상담 대화":      "#4C8BF5",
    "일상 대화":      "#34A853",
    "대출 사기형":    "#FBBC05",
    "수사기관 사칭형": "#EA4335",
}
MODEL_DISPLAY = {
    "mamba":     "RoBERTa-Mamba",
    "mamba_w32": "RoBERTa-Mamba (W32)",
    "gru":       "RoBERTa-GRU",
    "lstm":      "RoBERTa-LSTM",
    "avgpool":   "RoBERTa-AvgPool",
}

plt.rcParams["axes.unicode_minus"] = False

BG       = "#FFFFFF"
CARD_BG  = "#F7F8FA"
BORDER   = "#CCCCCC"
TEXT_FG  = "#222222"
TEXT_DIM = "#666666"


# ── 모델 추론 ──────────────────────────────────────────────────────────────────
def get_temperature(k, warmup=8, max_temp=4.0):
    if k >= warmup:
        return 1.0
    return max_temp - (max_temp - 1.0) * (k - 1) / (warmup - 1)


def load_and_run(key: str, text: str, temp_warmup: int, temp_max: float):
    exp_name = MODEL_REGISTRY[key]
    exp_dir  = PROJECT_ROOT / "models" / "classifier" / "experiments" / exp_name
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    for mod in ["config", "model", "dataset"]:
        sys.modules.pop(mod, None)

    from config import CHECKPOINT_DIR, DEVICE, IDX_TO_LABEL, LABELS, ENCODER_CONFIG, MAX_SEQ_LEN
    from model import build_model
    from dataset import build_segments
    from transformers import AutoTokenizer

    meta      = CHECKPOINT_DIR / f"{exp_name}_latest.json"
    ckpt_path = Path(json.loads(meta.read_text(encoding="utf-8"))["checkpoint_path"])
    model     = build_model().to(DEVICE)
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(
        ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])
    segments  = build_segments(tokenizer, text)
    n_windows = len(segments)
    results   = []

    with torch.no_grad():
        for k in range(1, n_windows + 1):
            segs    = segments[:k]
            max_len = min(max(len(s["input_ids"]) for s in segs), MAX_SEQ_LEN)
            S       = k
            input_ids      = torch.zeros(1, S, max_len, dtype=torch.long)
            attention_mask = torch.zeros(1, S, max_len, dtype=torch.long)
            segment_mask   = torch.ones(1, S, dtype=torch.bool)
            for j, seg in enumerate(segs):
                ids  = seg["input_ids"][:max_len]
                attn = seg["attention_mask"][:max_len]
                L    = len(ids)
                input_ids[0, j, :L]      = torch.tensor(ids,  dtype=torch.long)
                attention_mask[0, j, :L] = torch.tensor(attn, dtype=torch.long)

            out   = model(**{
                "input_ids":      input_ids.to(DEVICE),
                "attention_mask": attention_mask.to(DEVICE),
                "segment_mask":   segment_mask.to(DEVICE),
                "num_segments":   torch.tensor([S], dtype=torch.long).to(DEVICE),
            })
            temp  = get_temperature(k, temp_warmup, temp_max)
            probs = F.softmax(out["logits"].float() / temp, dim=-1)[0].cpu()
            results.append({
                "label": IDX_TO_LABEL[probs.argmax().item()],
                "conf":  probs.max().item(),
                "probs": {IDX_TO_LABEL[i]: probs[i].item() for i in range(len(LABELS))},
            })

    return results, n_windows


# ── 시각화 ─────────────────────────────────────────────────────────────────────
def visualize(model_keys, all_results, n_windows, true_label, output_path):
    n_models = len(model_keys)
    windows  = list(range(1, n_windows + 1))

    fig_h = max(6, n_windows * 0.72 + 3)
    fig   = plt.figure(figsize=(6.5 * n_models, fig_h))
    fig.patch.set_facecolor(BG)

    gs = GridSpec(1, n_models, figure=fig, wspace=0.08)

    for col, (key, results) in enumerate(zip(model_keys, all_results)):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        y_pos = np.arange(n_windows)  # 위→아래: window 1이 맨 위

        # ── 누적 가로 막대 ───────────────────────────────────────────────────
        lefts = np.zeros(n_windows)
        for lbl in LABELS_ORDER:
            vals  = np.array([results[w]["probs"][lbl] for w in range(n_windows)])
            color = CLASS_COLORS[lbl]
            bars  = ax.barh(y_pos, vals, left=lefts, color=color,
                            height=0.72, alpha=0.88, zorder=3)
            # 막대 위에 확률값 표시 (5% 이상만)
            for i, (v, l) in enumerate(zip(vals, lefts)):
                if v >= 0.07:
                    ax.text(l + v / 2, i, f"{v:.0%}",
                            ha="center", va="center",
                            color="white", fontsize=10,
                            fontweight="bold", zorder=5)
            lefts += vals


        # ── warmup 구간 배경 ─────────────────────────────────────────────────
        warmup_end = min(8, n_windows)
        ax.axhspan(-0.5, warmup_end - 0.5, alpha=0.08, color="#AAAAAA", zorder=0)

        # ── 축 설정 ──────────────────────────────────────────────────────────
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.5, n_windows - 0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"W{w}" for w in windows],
                           color=TEXT_FG, fontsize=12)
        ax.invert_yaxis()   # W1이 맨 위
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="x", colors=TEXT_DIM, labelsize=11)
        ax.tick_params(axis="y", colors=TEXT_FG)
        ax.grid(axis="x", color=BORDER, linewidth=0.6,
                linestyle="--", alpha=0.6, zorder=1)

        # ── 모델 제목 ────────────────────────────────────────────────────────
        final_label = results[-1]["label"]
        final_conf  = results[-1]["conf"]
        final_color = CLASS_COLORS.get(final_label, "white")

        if true_label:
            is_correct   = (final_label == true_label)
            verdict      = "정답" if is_correct else "오답"
            verdict_color = "#34A853" if is_correct else "#EA4335"
            title_str    = (f"{MODEL_DISPLAY.get(key, key.upper())}\n"
                            f"{final_label}  {final_conf:.1%}  "
                            f"[{verdict}]")
        else:
            verdict_color = final_color
            title_str     = (f"{MODEL_DISPLAY.get(key, key.upper())}\n"
                             f"{final_label}  {final_conf:.1%}")

        ax.set_title(title_str, color="#111111", fontsize=13,
                     fontweight="bold", pad=10,
                     bbox=dict(boxstyle="round,pad=0.5",
                               facecolor=verdict_color + "28",
                               edgecolor=verdict_color, linewidth=1.8))

        # 왼쪽 열에만 y축 레이블 표시, 나머지는 숨김
        if col > 0:
            ax.set_yticklabels([])

    # ── 공통 범례 ─────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[lbl], label=lbl, alpha=0.88)
        for lbl in LABELS_ORDER
    ]
    fig.legend(handles=legend_patches,
               loc="lower center", ncol=4,
               framealpha=0.25, facecolor=CARD_BG,
               edgecolor=BORDER, labelcolor="#111111",
               fontsize=12, bbox_to_anchor=(0.5, -0.04))

    # ── 전체 제목 ─────────────────────────────────────────────────────────────
    suptitle = "스트리밍 추론 모델 비교  —  윈도우별 4-class 확률 분포"
    if true_label:
        suptitle += f"   |   정답: {true_label}"
    fig.suptitle(suptitle, color="#111111", fontsize=15,
                 fontweight="bold", y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[저장] {output_path}")


# ── 진입점 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",        required=True)
    parser.add_argument("--models",      nargs="+", default=["mamba", "gru", "lstm"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--true-label",  default=None)
    parser.add_argument("--output",      default="streaming_comparison.png")
    parser.add_argument("--temp-warmup", type=int,   default=8)
    parser.add_argument("--temp-max",    type=float, default=4.0)
    args = parser.parse_args()

    all_results = []
    n_windows   = 0

    for key in args.models:
        print(f"[추론] {MODEL_DISPLAY.get(key, key)} ...", flush=True)
        results, n_win = load_and_run(key, args.text, args.temp_warmup, args.temp_max)
        all_results.append(results)
        n_windows = n_win
        print(f"       완료 (윈도우 {n_win}개)")

    visualize(args.models, all_results, n_windows, args.true_label, args.output)


if __name__ == "__main__":
    main()