"""
디코더 실험 F1 히트맵 시각화
대상: RoBERTa + {Mamba, GRU, LSTM}
"""

import json
import numpy as np
import koreanize_matplotlib  # noqa: F401  — registers NanumGothic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── 데이터 ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

EXPERIMENTS = {
    "Mamba": BASE / "roberta_mamba_freeze_init_4class/logs/roberta_mamba_freeze_init_4class_20260517_174922_test_20260517_185824_eval.json",
    "GRU":   BASE / "roberta_gru_freeze_init_4class/logs/roberta_gru_freeze_init_4class_20260517_174807_test_20260517_185755_eval.json",
    "LSTM":  BASE / "roberta_lstm_freeze_init_4class/logs/roberta_lstm_freeze_init_4class_20260517_174831_test_20260517_185818_eval.json",
}

METRICS = ["precision", "recall", "f1"]
METRIC_LABELS = ["Precision", "Recall", "F1"]

# ── 데이터 로드 ───────────────────────────────────────────────────────────
decoders = list(EXPERIMENTS.keys())
eval_data = {}
for name, path in EXPERIMENTS.items():
    with open(path) as f:
        eval_data[name] = json.load(f)

classes = eval_data[decoders[0]]["labels"]

plt.rcParams["axes.unicode_minus"] = False

# ── 히트맵 데이터 구성 (클래스 × 디코더, 메트릭별) ──────────────────────
# shape: (n_metrics, n_classes, n_decoders)
data = np.zeros((len(METRICS), len(classes), len(decoders)))

for j, decoder in enumerate(decoders):
    per_class = eval_data[decoder]["per_class"]
    for i, cls in enumerate(classes):
        for k, metric in enumerate(METRICS):
            data[k, i, j] = per_class[cls][metric]

# macro F1 (마지막 행에 추가)
macro_row = np.array([[eval_data[d]["macro_f1"] for d in decoders]])  # (1, n_decoders)

# ── Figure 구성: 3개 서브플롯 (Precision / Recall / F1) ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Decoder Comparison: Per-class Metrics\n(RoBERTa + Mamba / GRU / LSTM)",
             fontsize=13, fontweight="bold", y=1.02)

VMIN, VMAX = 0.96, 1.00
CMAP = mcolors.LinearSegmentedColormap.from_list("custom_blue", ["#e8f3ff", "#194aa6"])

for k, (ax, metric_label) in enumerate(zip(axes, METRIC_LABELS)):
    # 클래스 행 + macro 행
    mat = np.vstack([data[k], macro_row])          # (n_classes+1, n_decoders)
    row_labels = classes + ["Macro avg"]

    im = ax.imshow(mat, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")

    # 축 레이블
    ax.set_xticks(range(len(decoders)))
    ax.set_xticklabels(decoders, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels if k == 0 else [], fontsize=10)

    # 구분선 (macro avg 위)
    ax.axhline(len(classes) - 0.5, color="gray", linewidth=1.5, linestyle="--")

    # 셀 값 표기
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "black" if val < 0.985 else "white"
            # macro avg 행은 굵게
            weight = "bold" if i == len(classes) else "normal"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight=weight)

    ax.set_title(metric_label, fontsize=11, pad=8)

    # 컬러바 (마지막 서브플롯에만)
    if k == len(METRICS) - 1:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

plt.tight_layout()

out_path = BASE / "decoder_f1_heatmap.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장 완료: {out_path}")
plt.show()
