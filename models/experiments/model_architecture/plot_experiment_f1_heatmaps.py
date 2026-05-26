"""
5개 실험에 대한 클래스별 F1 히트맵 시각화
실험 보고서(2026-05-18_experiment_report.md) 기준 데이터

구조: rows=클래스, cols=비교 모델 (Macro F1은 열 레이블에 표기)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import os
import koreanize_matplotlib  # noqa: F401

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 저장 경로 ────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 공통 설정 ────────────────────────────────────────────────
CLASS_NAMES = ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]
CMAP = mcolors.LinearSegmentedColormap.from_list("custom_blue", ["#e8f3ff", "#194aa6"])
VMIN = 0.960
VMAX = 1.000
FONTSIZE_CELL  = 10
FONTSIZE_LABEL = 10
FONTSIZE_TITLE = 13


def make_col_labels(model_names, macro_f1s):
    """모델명 + Macro F1 점수를 x축 레이블로 생성."""
    return [f"{name}\n(F1: {f1:.4f})" for name, f1 in zip(model_names, macro_f1s)]


def draw_heatmap(ax, matrix, row_labels, col_labels):
    """
    matrix: (n_classes, n_models)
    rows   = 클래스,  cols = 모델
    """
    n_rows, n_cols = matrix.shape

    im = ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")

    for r in range(n_rows):
        for c in range(n_cols):
            val = matrix[r, c]
            color = "black" if val < 0.985 else "white"
            ax.text(c, r, f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=FONTSIZE_CELL,
                    color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=FONTSIZE_LABEL)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=FONTSIZE_LABEL)

    return im


def save_fig(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 실험 1. Mamba Layer 개수
# ══════════════════════════════════════════════════════════════
def plot_exp1_mamba_layers():
    # data[model, class]: 상담 대화, 일상 대화, 대출 사기형, 수사기관 사칭형
    data = np.array([
        [0.9975, 1.0000, 0.9876, 0.9899],  # L1
        [1.0000, 1.0000, 0.9876, 0.9874],  # L2 (기본)
    ])
    macro_f1s = [0.9938, 0.9937]
    model_names = ["L1", "L2 (기본)"]

    matrix = data.T  # (n_classes, n_models)
    col_labels = make_col_labels(model_names, macro_f1s)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = draw_heatmap(ax, matrix, CLASS_NAMES, col_labels)
    ax.set_title("Mamba Layer 개수별 F1\n(L4, L6: 학습 발산)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("Layer 수", fontsize=FONTSIZE_LABEL)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "exp1_mamba_layers_f1.png")


# ══════════════════════════════════════════════════════════════
# 실험 2. Local Encoder
# ══════════════════════════════════════════════════════════════
def plot_exp2_encoder():
    data = np.array([
        [1.0000, 1.0000, 0.9876, 0.9874],  # RoBERTa (기본)
        [0.9950, 0.9975, 0.9800, 0.9875],  # BERT
        [0.9899, 0.9975, 0.9777, 0.9850],  # KoBERT
        [0.9823, 0.9925, 0.9612, 0.9695],  # KoELECTRA
    ])
    macro_f1s = [0.9937, 0.9900, 0.9875, 0.9764]
    model_names = ["RoBERTa (기본)", "BERT", "KoBERT", "KoELECTRA"]

    matrix = data.T  # (n_classes, n_models)
    col_labels = make_col_labels(model_names, macro_f1s)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = draw_heatmap(ax, matrix, CLASS_NAMES, col_labels)
    ax.set_title("Local Encoder별 F1\n(Global Context Model: Mamba L2, d_state=16, w64)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("Local Encoder", fontsize=FONTSIZE_LABEL)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "exp2_encoder_f1.png")


# ══════════════════════════════════════════════════════════════
# 실험 3. Global Context Model
# ══════════════════════════════════════════════════════════════
def plot_exp3_decoder():
    data = np.array([
        [1.0000, 1.0000, 0.9876, 0.9874],  # Mamba (기본)
        [1.0000, 1.0000, 0.9851, 0.9848],  # GRU
        [0.9975, 1.0000, 0.9876, 0.9848],  # LSTM
    ])
    macro_f1s = [0.9937, 0.9925, 0.9925]
    model_names = ["Mamba (기본)", "GRU", "LSTM"]

    matrix = data.T  # (n_classes, n_models)
    col_labels = make_col_labels(model_names, macro_f1s)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = draw_heatmap(ax, matrix, CLASS_NAMES, col_labels)
    ax.set_title("Global Context Model별 F1\n(Local Encoder: RoBERTa, w64)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("Global Context Model", fontsize=FONTSIZE_LABEL)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "exp3_decoder_f1.png")


# ══════════════════════════════════════════════════════════════
# 실험 4. Mamba d_state
# ══════════════════════════════════════════════════════════════
def plot_exp4_dstate():
    data = np.array([
        [0.9975, 1.0000, 0.9876, 0.9899],  # d_state=16 (기본)
        [1.0000, 1.0000, 0.9851, 0.9848],  # d_state=32
        [1.0000, 1.0000, 0.9901, 0.9899],  # d_state=64
    ])
    macro_f1s = [0.9938, 0.9925, 0.9950]
    model_names = ["d_state=16 (기본)", "d_state=32", "d_state=64"]

    matrix = data.T  # (n_classes, n_models)
    col_labels = make_col_labels(model_names, macro_f1s)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = draw_heatmap(ax, matrix, CLASS_NAMES, col_labels)
    ax.set_title("Mamba d_state별 F1\n(Local Encoder: RoBERTa, Global Context Model: Mamba L1, w64)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("d_state", fontsize=FONTSIZE_LABEL)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "exp4_dstate_f1.png")


# ══════════════════════════════════════════════════════════════
# 실험 5. 윈도우 크기
# ══════════════════════════════════════════════════════════════
def plot_exp5_window():
    data = np.array([
        [1.0000, 1.0000, 0.9876, 0.9874],  # w64 (기본)
        [0.9925, 1.0000, 0.9781, 0.9851],  # w32
    ])
    macro_f1s = [0.9937, 0.9889]
    model_names = ["w64 (기본)", "w32"]

    matrix = data.T  # (n_classes, n_models)
    col_labels = make_col_labels(model_names, macro_f1s)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = draw_heatmap(ax, matrix, CLASS_NAMES, col_labels)
    ax.set_title("윈도우 크기별 F1\n(Local Encoder: RoBERTa, Global Context Model: Mamba L2, d_state=16)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("윈도우 크기", fontsize=FONTSIZE_LABEL)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "exp5_window_f1.png")


# ══════════════════════════════════════════════════════════════
# 통합 Figure (5개 실험을 하나의 figure에)
# ══════════════════════════════════════════════════════════════
def plot_all_combined():
    experiments = [
        {
            "title": "Mamba Layer 개수별 F1\n(RoBERTa, d_state=16, w64)",
            "data": np.array([
                [0.9975, 1.0000, 0.9876, 0.9899],
                [1.0000, 1.0000, 0.9876, 0.9874],
            ]),
            "model_names": ["L1", "L2 (기본)"],
            "macro_f1s":   [0.9938, 0.9937],
            "xlabel": "Layer 수",
        },
        {
            "title": "Local Encoder별 F1\n(Global Context Model: Mamba L2, d_state=16, w64)",
            "data": np.array([
                [1.0000, 1.0000, 0.9876, 0.9874],
                [0.9950, 0.9975, 0.9800, 0.9875],
                [0.9899, 0.9975, 0.9777, 0.9850],
                [0.9823, 0.9925, 0.9612, 0.9695],
            ]),
            "model_names": ["RoBERTa (기본)", "BERT", "KoBERT", "KoELECTRA"],
            "macro_f1s":   [0.9937, 0.9900, 0.9875, 0.9764],
            "xlabel": "Local Encoder",
        },
        {
            "title": "Global Context Model별 F1\n(Local Encoder: RoBERTa, w64)",
            "data": np.array([
                [1.0000, 1.0000, 0.9876, 0.9874],
                [1.0000, 1.0000, 0.9851, 0.9848],
                [0.9975, 1.0000, 0.9876, 0.9848],
            ]),
            "model_names": ["Mamba (기본)", "GRU", "LSTM"],
            "macro_f1s":   [0.9937, 0.9925, 0.9925],
            "xlabel": "Global Context Model",
        },
        {
            "title": "Mamba d_state별 F1\n(Local Encoder: RoBERTa, Global Context Model: Mamba L1, w64)",
            "data": np.array([
                [0.9975, 1.0000, 0.9876, 0.9899],
                [1.0000, 1.0000, 0.9851, 0.9848],
                [1.0000, 1.0000, 0.9901, 0.9899],
            ]),
            "model_names": ["d_state=16 (기본)", "d_state=32", "d_state=64"],
            "macro_f1s":   [0.9938, 0.9925, 0.9950],
            "xlabel": "d_state",
        },
        {
            "title": "윈도우 크기별 F1\n(Local Encoder: RoBERTa, Global Context Model: Mamba L2, d_state=16)",
            "data": np.array([
                [1.0000, 1.0000, 0.9876, 0.9874],
                [0.9925, 1.0000, 0.9781, 0.9851],
            ]),
            "model_names": ["w64 (기본)", "w32"],
            "macro_f1s":   [0.9937, 0.9889],
            "xlabel": "윈도우 크기",
        },
    ]

    fig = plt.figure(figsize=(14, 22))
    fig.suptitle("Ablation Study: 클래스별 F1 히트맵 (RoBERTa + Mamba 기반)",
                 fontsize=15, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.6)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    for ax, exp in zip(axes, experiments):
        matrix = exp["data"].T  # (n_classes, n_models)
        col_labels = make_col_labels(exp["model_names"], exp["macro_f1s"])
        im = draw_heatmap(ax, matrix, CLASS_NAMES, col_labels)
        ax.set_title(exp["title"], fontsize=FONTSIZE_TITLE - 1, fontweight="bold", pad=8)
        ax.set_xlabel(exp["xlabel"], fontsize=FONTSIZE_LABEL - 1)

    # 공통 컬러바
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.88])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("F1 Score", fontsize=10)

    path = os.path.join(SAVE_DIR, "all_experiments_f1_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 단일 모델: RoBERTa + Mamba L2, d_state=16, w64
# ══════════════════════════════════════════════════════════════
def plot_roberta_mamba_l2_baseline():
    """RoBERTa + Mamba L2, d_state=16, w64 클래스별 F1·Precision·Recall 히트맵."""
    # rows = 클래스, cols = 지표
    data = np.array([
        # F1       Precision  Recall
        [1.0000,   1.0000,    1.0000],   # 상담 대화
        [1.0000,   1.0000,    1.0000],   # 일상 대화
        [0.9876,   0.9803,    0.9950],   # 대출 사기형
        [0.9874,   0.9949,    0.9800],   # 수사기관 사칭형
    ])
    row_labels = CLASS_NAMES
    col_labels = ["F1", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(data, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto")

    n_rows, n_cols = data.shape
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            color = "black" if val < 0.985 else "white"
            ax.text(c, r, f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=FONTSIZE_CELL,
                    color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=FONTSIZE_LABEL)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=FONTSIZE_LABEL)

    ax.set_title("RoBERTa + Mamba L2, d_state=16, w64\n클래스별 성능 (Macro F1: 0.9937)",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("지표", fontsize=FONTSIZE_LABEL)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "roberta_mamba_l2_baseline_f1.png")


# ── 실행 ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== F1 히트맵 생성 시작 ===")
    plot_exp1_mamba_layers()
    plot_exp2_encoder()
    plot_exp3_decoder()
    plot_exp4_dstate()
    plot_exp5_window()
    plot_all_combined()
    plot_roberta_mamba_l2_baseline()
    print("=== 완료 ===")
