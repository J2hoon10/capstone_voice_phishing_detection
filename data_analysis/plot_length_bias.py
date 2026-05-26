"""
스크립트 길이 편향 시각화
- 일반 / 피싱 데이터의 길이 분포를 source 단위로 비교
- 저장 경로: figures/
"""

import csv
import statistics
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "val", "test"]
OUT_DIR = BASE  # figures/ 폴더 자체

# ── 데이터 로드 ─────────────────────────────────────────────────────────────
rows = []
for split in SPLITS:
    path = os.path.join(BASE, "..", f"{split}.csv")
    with open(path, encoding="utf-8-sig") as f:
        rows += list(csv.DictReader(f))

LABEL_NAMES = {
    "0": "일반(상담 A)",
    "1": "일반(일상 B)",
    "2": "피싱(대출)",
    "3": "피싱(수사기관)",
}

# source 매핑 (표시용)
SOURCE_DISPLAY = {
    "normal_original":    "원본",
    "normal_asr":         "원본+노이즈",
    "normal_asr_llm":     "LLM+노이즈",
    "normal_clean_llm":   "LLM+클린",
    "phishing_original":  "원본",
    "phishing_asr_llm":   "LLM+노이즈",
    "phishing_clean_llm": "LLM+클린",
}
SOURCE_COLORS = {
    "원본":       "#e15759",
    "원본+노이즈": "#f28e2b",
    "LLM+노이즈": "#4e79a7",
    "LLM+클린":  "#59a14f",
}

# ── 그룹화 ──────────────────────────────────────────────────────────────────
# {label: {display_source: [lengths]}}
data = {lbl: {} for lbl in LABEL_NAMES}
for r in rows:
    lbl = r.get("label", "").strip()
    src = r.get("source", "").strip()
    text = (r.get("text") or "").strip()
    if lbl not in data or not text:
        continue
    disp = SOURCE_DISPLAY.get(src, src)
    data[lbl].setdefault(disp, []).append(len(text))

plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — 전체 4클래스 KDE + Boxplot (상단 KDE, 하단 Box)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(20, 9),
                         gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("클래스별 스크립트 길이 분포 (전체 = train+val+test)", fontsize=15, fontweight="bold")

CLIP = 3000  # 시각화 범위

for col, lbl in enumerate(["0", "1", "2", "3"]):
    ax_kde = axes[0][col]
    ax_box = axes[1][col]
    title = LABEL_NAMES[lbl]

    src_data = data[lbl]
    all_lens = [x for v in src_data.values() for x in v]

    for disp_src, lens in sorted(src_data.items()):
        color = SOURCE_COLORS.get(disp_src, "#aaa")
        clipped = np.clip(lens, 0, CLIP)
        ax_kde.hist(clipped, bins=50, alpha=0.45, color=color,
                    label=f"{disp_src} (n={len(lens)})", density=True, edgecolor="none")

    # 중앙값 라인
    med = statistics.median(all_lens)
    ax_kde.axvline(min(med, CLIP), color="black", linestyle="--", linewidth=1.2,
                   label=f"중앙값 {int(med)}")
    ax_kde.set_title(title, fontsize=12)
    ax_kde.set_xlim(0, CLIP)
    ax_kde.set_xlabel("")
    ax_kde.set_ylabel("밀도" if col == 0 else "")
    ax_kde.legend(fontsize=7.5, loc="upper right")

    # Boxplot
    box_data = [np.clip(v, 0, CLIP) for v in src_data.values()]
    box_labels = list(src_data.keys())
    bp = ax_box.boxplot(box_data, vert=True, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, label in zip(bp["boxes"], box_labels):
        patch.set_facecolor(SOURCE_COLORS.get(label, "#aaa"))
        patch.set_alpha(0.7)
    ax_box.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax_box.set_ylim(0, CLIP)
    ax_box.set_ylabel("길이" if col == 0 else "")
    note = f"(>{CLIP} 클립됨)" if any(x > CLIP for v in src_data.values() for x in v) else ""
    ax_box.set_xlabel(note, fontsize=7, color="gray")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "fig1_length_dist_by_class.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"[저장] {out1}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — 원본 vs LLM 길이 비교 (피싱만)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("피싱 데이터: 원본 vs LLM 증강 길이 분포", fontsize=14, fontweight="bold")

CLIP2 = 5000
for col, lbl in enumerate(["2", "3"]):
    ax = axes[col]
    title = LABEL_NAMES[lbl]
    src_data = data[lbl]

    for disp_src in ["원본", "LLM+노이즈", "LLM+클린"]:
        lens = src_data.get(disp_src, [])
        if not lens:
            continue
        color = SOURCE_COLORS[disp_src]
        clipped = np.clip(lens, 0, CLIP2)
        ax.hist(clipped, bins=60, alpha=0.5, color=color,
                label=f"{disp_src}\n(n={len(lens)}, 평균={int(statistics.mean(lens))})",
                density=True, edgecolor="none")
        med = statistics.median(lens)
        ax.axvline(min(med, CLIP2), color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    orig_lens = src_data.get("원본", [])
    long_pct = sum(1 for x in orig_lens if x > 1200) / len(orig_lens) * 100 if orig_lens else 0
    ax.set_title(f"{title}\n원본 중 1,200자 초과: {long_pct:.1f}%", fontsize=11)
    ax.set_xlim(0, CLIP2)
    ax.set_xlabel("스크립트 길이 (자)")
    ax.set_ylabel("밀도" if col == 0 else "")
    ax.legend(fontsize=9)
    note = f"(>{CLIP2} 클립됨)" if any(x > CLIP2 for v in src_data.values() for x in v) else ""
    ax.text(0.99, 0.97, note, ha="right", va="top", transform=ax.transAxes,
            fontsize=7, color="gray")

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "fig2_phishing_orig_vs_llm.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"[저장] {out2}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — 일반 데이터: 원본(noised) vs LLM 비교
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("일반 데이터: 원본+노이즈 vs LLM 증강 길이 분포", fontsize=14, fontweight="bold")

CLIP3 = 3000
for col, lbl in enumerate(["0", "1"]):
    ax = axes[col]
    title = LABEL_NAMES[lbl]
    src_data = data[lbl]

    for disp_src in ["원본+노이즈", "LLM+노이즈", "LLM+클린"]:
        lens = src_data.get(disp_src, [])
        if not lens:
            continue
        color = SOURCE_COLORS[disp_src]
        clipped = np.clip(lens, 0, CLIP3)
        ax.hist(clipped, bins=50, alpha=0.5, color=color,
                label=f"{disp_src}\n(n={len(lens)}, 평균={int(statistics.mean(lens))})",
                density=True, edgecolor="none")
        med = statistics.median(lens)
        ax.axvline(min(med, CLIP3), color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, CLIP3)
    ax.set_xlabel("스크립트 길이 (자)")
    ax.set_ylabel("밀도" if col == 0 else "")
    ax.legend(fontsize=9)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "fig3_normal_orig_vs_llm.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"[저장] {out3}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — 4클래스 전체 길이 분포 비교 (단일 플롯, 클래스별 색)
# ════════════════════════════════════════════════════════════════════════════
CLASS_COLORS = {
    "0": "#4e79a7",
    "1": "#59a14f",
    "2": "#e15759",
    "3": "#f28e2b",
}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("4클래스 전체 길이 분포 비교", fontsize=14, fontweight="bold")

CLIP4 = 3000
ax1, ax2 = axes

# 히스토그램
for lbl in ["0","1","2","3"]:
    all_lens = np.clip([x for v in data[lbl].values() for x in v], 0, CLIP4)
    med = float(np.median(all_lens))
    ax1.hist(all_lens, bins=60, alpha=0.45, color=CLASS_COLORS[lbl],
             label=f"{LABEL_NAMES[lbl]} (중앙={int(med)})", density=True, edgecolor="none")
ax1.set_xlim(0, CLIP4)
ax1.set_xlabel("스크립트 길이 (자)")
ax1.set_ylabel("밀도")
ax1.set_title("히스토그램 (밀도)")
ax1.legend(fontsize=9)

# Boxplot (horizontal)
box_data = [np.clip([x for v in data[lbl].values() for x in v], 0, CLIP4) for lbl in ["0","1","2","3"]]
box_tick_labels = [LABEL_NAMES[l] for l in ["0","1","2","3"]]
bp = ax2.boxplot(box_data, vert=False, patch_artist=True,
                 medianprops=dict(color="black", linewidth=1.5),
                 flierprops=dict(marker=".", markersize=2, alpha=0.3))
for patch, lbl in zip(bp["boxes"], ["0","1","2","3"]):
    patch.set_facecolor(CLASS_COLORS[lbl])
    patch.set_alpha(0.7)
ax2.set_yticklabels(box_tick_labels, fontsize=10)
ax2.set_xlim(0, CLIP4)
ax2.set_xlabel("스크립트 길이 (자)")
ax2.set_title("Boxplot 비교")
note = f"(>{CLIP4} 클립됨)"
ax2.text(0.99, 0.02, note, ha="right", va="bottom", transform=ax2.transAxes,
         fontsize=7, color="gray")

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "fig4_4class_comparison.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"[저장] {out4}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — 구간별 비율 누적 막대
# ════════════════════════════════════════════════════════════════════════════
buckets = [(0,100),(100,200),(200,300),(300,500),(500,1000),(1000,2000),(2000,99999)]
bucket_labels_disp = ["0-99","100-199","200-299","300-499","500-999","1000-1999","2000+"]
BUCKET_COLORS = ["#aecbfa","#5c85d6","#f28e2b","#e15759","#76b7b2","#59a14f","#b07aa1"]

fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("클래스별 길이 구간 비율 (%)", fontsize=13, fontweight="bold")

x = np.arange(len(LABEL_NAMES))
bottom = np.zeros(len(LABEL_NAMES))

for bi, (lo, hi) in enumerate(buckets):
    proportions = []
    for lbl in ["0","1","2","3"]:
        lens = [x_ for v in data[lbl].values() for x_ in v]
        n = len(lens)
        cnt = sum(1 for x_ in lens if lo <= x_ < hi)
        proportions.append(cnt / n * 100 if n else 0)
    bars = ax.bar(x, proportions, bottom=bottom, color=BUCKET_COLORS[bi],
                  label=bucket_labels_disp[bi], alpha=0.85)
    # 5% 이상인 구간만 레이블 표시
    for xi, (prop, bot) in enumerate(zip(proportions, bottom)):
        if prop >= 5:
            ax.text(xi, bot + prop / 2, f"{prop:.0f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottom += np.array(proportions)

ax.set_xticks(x)
ax.set_xticklabels([LABEL_NAMES[l] for l in ["0","1","2","3"]], fontsize=11)
ax.set_ylabel("비율 (%)")
ax.set_ylim(0, 103)
ax.legend(title="길이 구간(자)", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
plt.tight_layout()
out5 = os.path.join(OUT_DIR, "fig5_length_bucket_ratio.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"[저장] {out5}")

print("\n[완료] 모든 그래프 저장됨")
