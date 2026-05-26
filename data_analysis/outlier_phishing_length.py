"""
outlier_phishing_length.py

피싱 카테고리(대출 사기형, 수사기관 사칭형)의 대화 길이(세그먼트 수) 통계 분석
→ IQR / Z-score 기반 아웃라이어 탐지 및 목록 출력
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA_DIR  = Path(__file__).resolve().parents[1] / "models" / "main" / "data_augmentation" / "output" / "4class"
CSV_FILES = [DATA_DIR / "train.csv", DATA_DIR / "val.csv", DATA_DIR / "test.csv"]

PHISHING_CATS  = {"대출 사기형", "수사기관 사칭형"}
NORMAL_CATS    = {"상담 대화", "일상 대화"}

IQR_MULTIPLIER = 1.5   # 표준 fence; 3.0 이면 extreme outlier
ZSCORE_THRESH  = 2.5


# ── helpers ───────────────────────────────────────────────────────────────────
def get_nseg(row):
    sr = row.get("segment_risks", "")
    if sr:
        try:
            return len(json.loads(sr))
        except Exception:
            pass
    return max(1, len((row.get("text") or "").split()) // 15)


def percentile(sorted_data, q):
    """0-100 사이 q에 대한 백분위 값 (선형 보간)."""
    n = len(sorted_data)
    if n == 0:
        return float("nan")
    idx = q / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def iqr_fence(sorted_data):
    q1 = percentile(sorted_data, 25)
    q3 = percentile(sorted_data, 75)
    iqr = q3 - q1
    upper = q3 + IQR_MULTIPLIER * iqr
    lower = q1 - IQR_MULTIPLIER * iqr
    return q1, q3, iqr, lower, upper


def zscore(value, mean, std):
    return (value - mean) / std if std > 0 else 0.0


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
all_rows = []
for csv_path in CSV_FILES:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat  = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            row["_category"] = cat
            row["_nseg"]     = get_nseg(row)
            row["_split"]    = csv_path.stem
            all_rows.append(row)


# ── 카테고리별 분류 ───────────────────────────────────────────────────────────
cat_rows = defaultdict(list)
for r in all_rows:
    cat_rows[r["_category"]].append(r)

ALL_CATS = ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전체 카테고리 길이 분포 요약
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  [1] 카테고리별 대화 길이(세그먼트 수) 분포 요약")
print("=" * 80)
print(f"\n  {'category':<20} {'N':>5}  {'mean':>6}  {'std':>6}  "
      f"{'p10':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'p90':>5} {'p95':>5} {'max':>5}")
print("  " + "-" * 85)

for cat in ALL_CATS:
    rows = cat_rows.get(cat, [])
    ns   = sorted(r["_nseg"] for r in rows)
    if not ns:
        continue
    mean = statistics.mean(ns)
    std  = statistics.pstdev(ns)
    p    = lambda q: percentile(ns, q)
    print(f"  {cat:<20} {len(ns):>5}  {mean:>6.1f}  {std:>6.1f}  "
          f"{p(10):>5.1f} {p(25):>5.1f} {p(50):>5.1f} {p(75):>5.1f} "
          f"{p(90):>5.1f} {p(95):>5.1f} {max(ns):>5}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. IQR / Z-score 아웃라이어 탐지 (피싱 카테고리)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"  [2] 피싱 카테고리 아웃라이어 탐지  "
      f"(IQR × {IQR_MULTIPLIER}  /  |Z| > {ZSCORE_THRESH})")
print("=" * 80)

outlier_rows = []   # 전체 split 포함 아웃라이어 수집

for cat in ["대출 사기형", "수사기관 사칭형"]:
    rows = cat_rows.get(cat, [])
    ns   = sorted(r["_nseg"] for r in rows)
    if not ns:
        continue

    mean = statistics.mean(ns)
    std  = statistics.pstdev(ns)
    q1, q3, iqr, lower_fence, upper_fence = iqr_fence(ns)

    print(f"\n  ▶ {cat}")
    print(f"    N={len(ns)},  mean={mean:.1f},  std={std:.1f},  "
          f"IQR={iqr:.1f},  Q1={q1:.1f},  Q3={q3:.1f}")
    print(f"    IQR upper fence : {upper_fence:.1f}  "
          f"(= Q3 + {IQR_MULTIPLIER} × IQR)")
    print(f"    Z-score |z| > {ZSCORE_THRESH}  → nseg > {mean + ZSCORE_THRESH * std:.1f}")

    # 아웃라이어 기준: IQR OR Z-score 중 하나라도 해당
    cat_outliers = []
    for r in rows:
        n  = r["_nseg"]
        z  = zscore(n, mean, std)
        is_iqr    = n > upper_fence
        is_zscore = abs(z) > ZSCORE_THRESH
        if is_iqr or is_zscore:
            cat_outliers.append((r, n, z, is_iqr, is_zscore))
    cat_outliers.sort(key=lambda x: -x[1])

    print(f"\n    아웃라이어 {len(cat_outliers)}개 (IQR 또는 |Z|>{ZSCORE_THRESH}):")
    print(f"    {'split':<6} {'nseg':>5}  {'z':>6}  {'IQR':^5}  {'Z':^5}  id")
    print("    " + "-" * 70)
    for r, n, z, is_iqr, is_z in cat_outliers:
        sid = (r.get("id") or "")[:50]
        print(f"    {r['_split']:<6} {n:>5}  {z:>6.2f}  "
              f"{'✓' if is_iqr else ' ':^5}  {'✓' if is_z else ' ':^5}  {sid}")
        outlier_rows.append(r)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 분할별 아웃라이어 수 요약 (제거 영향 파악)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  [3] 분할별 아웃라이어 수 (제거 시 영향)")
print("=" * 80)

split_count = defaultdict(lambda: defaultdict(int))
for r in outlier_rows:
    split_count[r["_split"]][r["_category"]] += 1

for split in ["train", "val", "test"]:
    counts = split_count.get(split, {})
    total  = sum(counts.values())
    detail = ", ".join(f"{cat}: {cnt}" for cat, cnt in sorted(counts.items()))
    print(f"  {split:<6}  {total:>3}개  ({detail})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 아웃라이어 제거 후 예상 Very Long 비율
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  [4] 아웃라이어 제거 후 Very Long (16+) 비율 변화 예측  (train 기준)")
print("=" * 80)

outlier_ids = {(r.get("id") or "") for r in outlier_rows if r["_split"] == "train"}

for cat in ALL_CATS:
    rows  = [r for r in cat_rows.get(cat, []) if r["_split"] == "train"]
    after = [r for r in rows if (r.get("id") or "") not in outlier_ids]
    if not rows:
        continue
    before_vl = sum(1 for r in rows  if r["_nseg"] >= 16)
    after_vl  = sum(1 for r in after if r["_nseg"] >= 16)
    print(f"  {cat:<20}  "
          f"제거 전 {before_vl}/{len(rows)} ({before_vl/len(rows)*100:.1f}%)  →  "
          f"제거 후 {after_vl}/{len(after)} ({after_vl/len(after)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 권장 조치
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  [5] 요약 및 권장 조치")
print("=" * 80)
print(f"""
  아웃라이어로 탐지된 train 샘플 : {sum(1 for r in outlier_rows if r["_split"]=="train")}개

  권장 전략:
    A) train 아웃라이어 제거 → Very Long 피싱 편향 완화
    B) 일상/상담 대화 연결 증강으로 Very Long 정상 대화 생성
    C) 제거 수만큼 증강 샘플 추가 → 전체 데이터 수 유지
""")
