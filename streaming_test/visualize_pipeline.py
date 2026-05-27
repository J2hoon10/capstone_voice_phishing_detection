"""
visualize_pipeline.py

파이프라인 속도 측정 결과 시각화 (PIPELINE_REPORT.md 기반)
  - Chart 1: RTF 비교 (설정별)
  - Chart 2: 처리 시간 분포 (STT vs 추론)
  - Chart 3: W64 vs W32 조기 탐지 타임라인

출력: pipeline_speed_summary.png
"""

import koreanize_matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── 측정 데이터 ────────────────────────────────────────────────────────────────
# (PIPELINE_REPORT.md 섹션 4.2 / 4.5 / 4.6 기준)

CONFIGS = [
    {"label": "30s+10s\nCPU",   "rtf": 0.341, "stt": 70.7, "inf": 0.9,  "color": "#AAAAAA"},
    {"label": "30s+10s\nGPU",   "rtf": 0.297, "stt": 60.9, "inf": 1.3,  "color": "#7BAFD4"},
    {"label": "15s+8s\nGPU / W64", "rtf": 0.067, "stt": 13.5, "inf": 0.6,  "color": "#4C8BF5"},
    {"label": "15s+8s\nGPU / W32", "rtf": 0.057, "stt": 11.3, "inf": 0.60, "color": "#34A853"},
]

# 조기 탐지 타임라인 데이터
AUDIO_DURATION = 209.9
TIMELINE = [
    {
        "model":         "W64  (WINDOW=64)",
        "first_window":  0,          # 첫 추론 시작 (초)
        "conf90":        60,         # 90% 확신 도달 청크 시작 (초)
        "conf90_end":    75,         # 90% 확신 도달 청크 끝 (초)
        "color":         "#4C8BF5",
    },
    {
        "model":         "W32  (WINDOW=32)",
        "first_window":  0,
        "conf90":        30,
        "conf90_end":    45,
        "color":         "#34A853",
    },
]

# ── 스타일 상수 ────────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
CARD_BG  = "#F7F8FA"
BORDER   = "#DDDDDD"
TEXT     = "#222222"
TEXT_DIM = "#888888"
ACCENT   = "#EA4335"

plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.facecolor":     CARD_BG,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    TEXT,
    "xtick.color":        TEXT,
    "ytick.color":        TEXT,
    "grid.color":         BORDER,
    "grid.linewidth":     0.8,
    "grid.linestyle":     "--",
    "figure.facecolor":   BG,
    "font.size":          15,
})


def draw_rtf_chart(ax):
    """Chart 1: RTF 비교 막대 그래프"""
    labels = [c["label"] for c in CONFIGS]
    rtfs   = [c["rtf"]   for c in CONFIGS]
    colors = [c["color"] for c in CONFIGS]

    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, rtfs, color=colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.8)

    # RTF=1.0 기준선
    ax.axhline(1.0, color=ACCENT, linewidth=1.6, linestyle="--", zorder=4, label="RTF = 1.0 (실시간)")

    baseline_rtf = rtfs[0]  # 30s CPU 기준

    # 막대 위: RTF 수치 + CPU 대비 배속 뱃지
    for i, (bar, rtf) in enumerate(zip(bars, rtfs)):
        cx = bar.get_x() + bar.get_width() / 2
        top = bar.get_height()

        # RTF 수치
        ax.text(cx, top + 0.009, f"{rtf:.3f}x",
                ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=TEXT)

        # CPU 대비 배속 뱃지 (기준 막대 제외) — 막대 위 RTF 수치 아래에 컬러 박스로 표시
        if i > 0:
            speedup = baseline_rtf / rtf
            ax.text(cx, top + 0.068,
                    f"▲ CPU 대비\n{speedup:.1f}배",
                    ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=CONFIGS[i]["color"],
                              edgecolor="none", alpha=0.88))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("RTF (낮을수록 빠름)", fontsize=15, color=TEXT)
    ax.set_ylim(0, 0.58)
    ax.set_title("설정별 RTF 비교", fontsize=17, fontweight="bold", color=TEXT, pad=10)
    ax.legend(fontsize=17, framealpha=0.6)
    ax.grid(axis="y", zorder=0)
    ax.tick_params(bottom=False)


def draw_time_breakdown(ax):
    """Chart 2: 처리 시간 분포 (누적 가로 막대)"""
    labels = [c["label"].replace("\n", " ") for c in CONFIGS]
    stts   = [c["stt"] for c in CONFIGS]
    infs   = [c["inf"] for c in CONFIGS]
    totals = [s + i for s, i in zip(stts, infs)]

    y = np.arange(len(CONFIGS))
    h = 0.5

    stt_bars = ax.barh(y, stts,           height=h, color="#7BAFD4", zorder=3, label="STT (Whisper)")
    inf_bars = ax.barh(y, infs, left=stts, height=h, color="#FBBC05", zorder=3, label="추론 (분류기)")

    # 비율 레이블
    for i, (s, inf, tot) in enumerate(zip(stts, infs, totals)):
        stt_pct = s / tot * 100
        inf_pct = inf / tot * 100
        ax.text(s / 2, i, f"{stt_pct:.0f}%",
                ha="center", va="center", fontsize=17, color="white", fontweight="bold")
        if inf_pct >= 6:
            ax.text(s + inf / 2, i, f"{inf_pct:.0f}%",
                    ha="center", va="center", fontsize=17, color=TEXT, fontweight="bold")

        # 총 처리 시간 우측 표시
        ax.text(tot + 0.8, i, f"{tot:.1f}s",
                va="center", fontsize=17, color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("처리 시간 (초)", fontsize=15, color=TEXT)
    ax.set_title("처리 시간 분포 (STT vs 추론)", fontsize=17, fontweight="bold", color=TEXT, pad=10)
    ax.legend(fontsize=17, framealpha=0.6, loc="lower right")
    ax.grid(axis="x", zorder=0)
    ax.set_xlim(0, max(totals) * 1.15)
    ax.invert_yaxis()


def draw_detection_timeline(ax):
    """Chart 3: W64 vs W32 조기 탐지 타임라인"""
    ax.set_facecolor(BG)

    y_pos   = [1.0, 0.0]
    bar_h   = 0.28

    for i, (entry, y) in enumerate(zip(TIMELINE, y_pos)):
        color = entry["color"]

        # 전체 오디오 배경
        ax.barh(y, AUDIO_DURATION, height=bar_h,
                color="#EEEEEE", edgecolor=BORDER, linewidth=0.8, zorder=2)

        # 탐지 시작 ~ 90% 확신 구간
        ax.barh(y, entry["conf90_end"], height=bar_h,
                color=color, alpha=0.25, zorder=3)

        # 90% 확신 도달 청크
        width = entry["conf90_end"] - entry["conf90"]
        ax.barh(y, width, left=entry["conf90"], height=bar_h,
                color=color, alpha=0.9, zorder=4,
                label=f"{entry['model']} — 90% 확신 도달")

        # 첫 추론 마커
        ax.plot(entry["first_window"] + 1, y, marker="D",
                color=color, markersize=7, zorder=6)

        # 90% 확신 시점 수직선
        ax.axvline(entry["conf90"], color=color, linewidth=1.5,
                   linestyle=":", alpha=0.7, zorder=5)
        ax.text(entry["conf90"] + 2, y + bar_h / 2 + 0.04,
                f"{entry['conf90']}s\n90%↑",
                fontsize=17, color=color, fontweight="bold", va="bottom")

        # 모델 레이블
        ax.text(-5, y, entry["model"],
                ha="right", va="center", fontsize=15,
                color=color, fontweight="bold")

    # 오디오 전체 길이 표시
    ax.axvline(AUDIO_DURATION, color=TEXT_DIM, linewidth=1.2,
               linestyle="-.", alpha=0.5)
    ax.text(AUDIO_DURATION + 2, 0.5, f"총 {AUDIO_DURATION:.0f}s",
            fontsize=17, color=TEXT_DIM, va="center")

    # 조기 탐지 이득 브라켓
    y_mid = 0.5
    ax.annotate("", xy=(30, y_mid), xytext=(60, y_mid),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6))
    ax.text(45, y_mid + 0.12, "30초 빠름",
            ha="center", fontsize=15, color=ACCENT, fontweight="bold")

    ax.set_xlim(-30, AUDIO_DURATION + 20)
    ax.set_ylim(-0.35, 1.5)
    ax.set_xlabel("통화 경과 시간 (초)", fontsize=15, color=TEXT)
    ax.set_title("W64 vs W32  —  조기 탐지 타임라인 (15초 청크 + 8초 오버랩)",
                 fontsize=17, fontweight="bold", color=TEXT, pad=10)

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", zorder=0)

    legend_patches = [
        mpatches.Patch(color="#4C8BF5", alpha=0.85, label="W64  90% 확신 구간"),
        mpatches.Patch(color="#34A853", alpha=0.85, label="W32  90% 확신 구간"),
        plt.Line2D([0], [0], marker="D", color="#888888",
                   markersize=9, linestyle="none", label="첫 추론 시작"),
    ]
    ax.legend(handles=legend_patches, fontsize=17, framealpha=0.6,
              loc="upper right")


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(17, 13), facecolor=BG)
    fig.subplots_adjust(hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    draw_rtf_chart(ax1)
    draw_time_breakdown(ax2)
    draw_detection_timeline(ax3)

    fig.suptitle("실시간 음성 피싱 탐지 파이프라인  —  속도 측정 결과 요약",
                 fontsize=19, fontweight="bold", color=TEXT, y=1.01)

    out = Path(__file__).parent / "figure" / "pipeline_speed_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
