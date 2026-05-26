"""
analyze_baro.py

1. '바로 이 목소리' 데이터 중 대출 관련 언어가 섞인 샘플 목록화
2. Very Long(16+) 세그먼트 샘플들의 대화 길이 분포
3. 일반 대화(상담/일상) 길이 분포 비교
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[3] / "models" / "classifier" / "preprocessing" / "output" / "4class"
CSV_FILES = [DATA_DIR / "train.csv", DATA_DIR / "val.csv", DATA_DIR / "test.csv"]

# 대출 관련 키워드
LOAN_KEYWORDS = [
    "대출", "신용", "금리", "한도", "이자", "대환", "저금리", "담보",
    "원리금", "상환", "연체", "신용등급", "신용점수", "상향", "하향",
    "대출금", "대출상담", "대출상담실", "대출실", "햇살론", "카카오뱅크",
    "케이뱅크", "새마을금고", "저축은행", "캐피탈", "파이낸셜",
]

VERY_LONG_MIN = 16

# ── 세그먼트 수 계산 (segment_risks 길이 기반) ────────────────────────────────
def get_nseg(row):
    sr = row.get("segment_risks", "")
    if sr:
        try:
            return len(json.loads(sr))
        except Exception:
            pass
    # fallback: 토큰 수 추정 (텍스트 길이 / 15 정도)
    return max(1, len((row.get("text") or "").split()) // 15)


def contains_loan(text):
    return [kw for kw in LOAN_KEYWORDS if kw in text]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
all_rows = []
for csv_path in CSV_FILES:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            row["_category_raw"] = cat
            row["_category_norm"] = cat
            # ID에 '바로 이 목소리'가 포함된 샘플이 해당 원본 데이터
            row["_is_baro"] = "바로 이 목소리" in (row.get("id") or "")
            row["_nseg"] = get_nseg(row)
            row["_split"] = csv_path.stem
            all_rows.append(row)

# ── 1. 바로 이 목소리 중 대출 언어 포함 샘플 ─────────────────────────────────
print("\n" + "=" * 80)
print("  [1] '바로 이 목소리' 데이터 중 대출 관련 언어 포함 샘플")
print("=" * 80)

baro_rows = [r for r in all_rows if r["_is_baro"]]
baro_loan = [(r, contains_loan(r["text"])) for r in baro_rows]
baro_loan_hit = [(r, kws) for r, kws in baro_loan if kws]

print(f"\n  전체 '바로 이 목소리' 샘플: {len(baro_rows)}개")
print(f"  대출 언어 포함 샘플: {len(baro_loan_hit)}개 ({len(baro_loan_hit)/len(baro_rows)*100:.1f}%)\n")

print(f"  {'split':<6} {'id':<45} {'nseg':>5}  {'키워드'}")
print("  " + "-" * 78)
for r, kws in sorted(baro_loan_hit, key=lambda x: -x[0]["_nseg"]):
    sid = (r.get("id") or "")[:44]
    print(f"  {r['_split']:<6} {sid:<45} {r['_nseg']:>5}  {', '.join(kws[:5])}")

# ── 2. Very Long 구간 길이 분포 ───────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"  [2] Very Long ({VERY_LONG_MIN}+ 세그먼트) 샘플 길이 분포  (전체 split 합산)")
print("=" * 80)

very_long = [r for r in all_rows if r["_nseg"] >= VERY_LONG_MIN]
print(f"\n  총 {len(very_long)}개\n")

print(f"  {'category':<20} {'count':>6}  세그먼트 분포 (min/mean/max)")
print("  " + "-" * 60)
by_cat = defaultdict(list)
for r in very_long:
    by_cat[r["_category_norm"]].append(r["_nseg"])

for cat, nsegs in sorted(by_cat.items()):
    mn, mx, avg = min(nsegs), max(nsegs), sum(nsegs)/len(nsegs)
    print(f"  {cat:<20} {len(nsegs):>6}  {mn} / {avg:.1f} / {mx}")

print(f"\n  세그먼트 수별 상세:")
print(f"  {'nseg':>6}  {'category':<20}  {'split':<6}  id")
print("  " + "-" * 70)
for r in sorted(very_long, key=lambda x: -x["_nseg"]):
    sid = (r.get("id") or "")[:40]
    print(f"  {r['_nseg']:>6}  {r['_category_norm']:<20}  {r['_split']:<6}  {sid}")

# ── 3. 일반 대화 vs 피싱 길이 비교 ───────────────────────────────────────────
print("\n" + "=" * 80)
print("  [3] 카테고리별 대화 길이 통계 (전체 split)")
print("=" * 80)

cat_nsegs = defaultdict(list)
for r in all_rows:
    cat_nsegs[r["_category_norm"]].append(r["_nseg"])

print(f"\n  {'category':<20} {'count':>6}  {'min':>5}  {'p25':>5}  {'p50':>5}  {'p75':>5}  {'p90':>5}  {'max':>5}")
print("  " + "-" * 72)
import statistics
for cat in ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]:
    ns = sorted(cat_nsegs.get(cat, []))
    if not ns:
        continue
    n = len(ns)
    p = lambda q: ns[int(q * n / 100)]
    print(f"  {cat:<20} {n:>6}  {min(ns):>5}  {p(25):>5}  {p(50):>5}  {p(75):>5}  {p(90):>5}  {max(ns):>5}")

print(f"\n  [16+ 세그먼트 비율 by category]")
print(f"  {'category':<20} {'16+ 비율':>10}")
print("  " + "-" * 32)
for cat in ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]:
    ns = cat_nsegs.get(cat, [])
    if not ns:
        continue
    ratio = sum(1 for x in ns if x >= VERY_LONG_MIN) / len(ns) * 100
    print(f"  {cat:<20} {ratio:>9.1f}%")
