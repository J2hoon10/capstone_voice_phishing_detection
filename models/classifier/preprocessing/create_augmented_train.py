"""
create_augmented_train.py

1. 피싱 카테고리에서 Z-score |z| > 2.5 아웃라이어를 train에서 제거
2. 일상 대화 샘플을 연결(concat)해 Very Long(16+) 정상 대화 증강
3. LLM 생성 증강 데이터 통합:
     - output/long_normal_preprocessed.csv  (긴 정상 대화)
     - output/short_phishing_preprocessed.csv (짧은 피싱 대화)
4. output/4class/train_augmented.csv 저장

전처리 선행 조건:
    python preprocess_augmented_data.py --mode clean-normal
    python preprocess_augmented_data.py --mode submit-phishing
    python preprocess_augmented_data.py --mode collect-phishing

사용법:
    python create_augmented_train.py [--target-ratio 0.15] [--seed 42] [--dry-run]
    python create_augmented_train.py [--no-llm-normal] [--no-llm-phishing]
"""

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "output" / "4class"
OUT_DIR   = Path(__file__).resolve().parent / "output"
INPUT_CSV  = DATA_DIR / "train.csv"
OUTPUT_CSV = DATA_DIR / "train_augmented.csv"

LLM_NORMAL_CSV   = OUT_DIR / "long_normal_preprocessed.csv"
LLM_PHISHING_CSV = OUT_DIR / "short_phishing_preprocessed.csv"

PHISHING_CATS  = {"대출 사기형", "수사기관 사칭형"}
PHISHING_LABELS = {"2", "3"}  # 4클래스 레이블 기준
AUGMENT_CAT   = "일상 대화"   # 증강 대상
AUGMENT_LABEL = "1"           # 일상 대화 label 값

ZSCORE_THRESH = 2.5
VERY_LONG_MIN = 16

SEP = " "   # 연결 구분자 (공백)


# ── helpers ───────────────────────────────────────────────────────────────────
def get_nseg(row):
    sr = row.get("segment_risks", "")
    if sr:
        try:
            return len(json.loads(sr))
        except Exception:
            pass
    return max(1, len((row.get("text") or "").split()) // 15)


def zscore(value, mean, std):
    return (value - mean) / std if std > 0 else 0.0


def concat_segment_risks(risk_lists):
    """여러 segment_risks JSON 배열을 이어붙임."""
    merged = []
    for rl in risk_lists:
        try:
            merged.extend(json.loads(rl))
        except Exception:
            pass
    return json.dumps(merged, ensure_ascii=False) if merged else ""


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat  = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            row["_category"] = cat
            row["_nseg"]     = get_nseg(row)
            rows.append(row)
    return rows


# ── Step 1. Z-score 아웃라이어 탐지 ──────────────────────────────────────────
def detect_outliers(rows):
    """피싱 카테고리 내 |z| > ZSCORE_THRESH 샘플 ID 집합 반환."""
    cat_nsegs = defaultdict(list)
    for r in rows:
        if r["_category"] in PHISHING_CATS:
            cat_nsegs[r["_category"]].append(r["_nseg"])

    outlier_ids = set()
    print(f"\n[Z-score 아웃라이어 탐지]  |z| > {ZSCORE_THRESH}")
    for cat, nsegs in cat_nsegs.items():
        mean = statistics.mean(nsegs)
        std  = statistics.pstdev(nsegs)
        threshold = mean + ZSCORE_THRESH * std
        print(f"  {cat}: mean={mean:.1f}, std={std:.1f}, 상한={threshold:.1f}")
        for r in rows:
            if r["_category"] == cat and r["_nseg"] > threshold:
                outlier_ids.add(r.get("id") or r.get("\ufeffid") or "")
                print(f"    제거: {(r.get('id') or r.get(chr(65279)+'id','')):<45} nseg={r['_nseg']:>4}  z={zscore(r['_nseg'],mean,std):>5.2f}")
    print(f"  → 총 {len(outlier_ids)}개 제거 예정\n")
    return outlier_ids


# ── Step 2. 일상 대화 연결 증강 ────────────────────────────────────────────────
def augment_normal(rows, target_ratio, seed):
    """
    일상 대화 샘플 중 짧은 것(1~8 세그먼트)을 랜덤으로 이어붙여
    Very Long(16+ 세그먼트) 샘플을 생성.
    target_ratio: 증강 후 일상 대화 내 16+ 비율 목표
    """
    rng = random.Random(seed)

    normal_rows = [r for r in rows if r["_category"] == AUGMENT_CAT]
    short_pool  = [r for r in normal_rows if r["_nseg"] <= 8]

    current_vl    = sum(1 for r in normal_rows if r["_nseg"] >= VERY_LONG_MIN)
    current_total = len(normal_rows)

    # 목표 Very Long 수: target_ratio = (current_vl + add) / (current_total + add)
    # → add = (target_ratio * current_total - current_vl) / (1 - target_ratio)
    need_add = int(
        (target_ratio * current_total - current_vl) / max(1 - target_ratio, 1e-9)
    )
    need_add = max(need_add, 0)

    print(f"[일상 대화 증강]")
    print(f"  현재 Very Long: {current_vl}/{current_total} ({current_vl/current_total*100:.1f}%)")
    print(f"  목표 비율: {target_ratio*100:.0f}%  →  {need_add}개 추가 필요")
    print(f"  연결 풀(short pool): {len(short_pool)}개\n")

    new_rows = []
    if len(short_pool) < 4:
        print("  ⚠ short_pool 부족, 증강 건너뜀")
        return new_rows

    for i in range(need_add):
        # 4~8개 샘플을 랜덤 선택해 이어붙임 (목표 세그먼트: 16~32)
        n_concat = rng.randint(4, 8)
        chosen   = rng.choices(short_pool, k=n_concat)

        merged_text  = SEP.join(r["text"] for r in chosen)
        merged_risks = concat_segment_risks([r.get("segment_risks", "") for r in chosen])
        merged_nseg  = sum(r["_nseg"] for r in chosen)

        new_row = {
            "\ufeffid"        : f"augmented_{AUGMENT_CAT}_concat_{i:04d}",
            "text"           : merged_text,
            "label"          : AUGMENT_LABEL,
            "category"       : AUGMENT_CAT,
            "source"         : "augmented_concat",
            "filename"       : "",
            "segment_risks"  : merged_risks,
            "_category"      : AUGMENT_CAT,
            "_nseg"          : merged_nseg,
        }
        new_rows.append(new_row)

    actual_vl = sum(1 for r in new_rows if r["_nseg"] >= VERY_LONG_MIN)
    print(f"  생성 완료: {len(new_rows)}개  (16+ 해당: {actual_vl}개)")
    total_after = current_total + len(new_rows)
    vl_after    = current_vl + actual_vl
    print(f"  증강 후 Very Long: {vl_after}/{total_after} ({vl_after/total_after*100:.1f}%)\n")
    return new_rows


# ── 저장 ─────────────────────────────────────────────────────────────────────
FIELDNAMES = ["\ufeffid", "text", "label", "category", "source", "filename", "segment_risks"]

def save_csv(rows, path):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[저장] {path}  ({len(rows)}행)")


# ── 분포 요약 출력 ─────────────────────────────────────────────────────────────
def print_summary(label, rows):
    cat_counts = defaultdict(lambda: {"total": 0, "vl": 0})
    for r in rows:
        cat = r["_category"]
        cat_counts[cat]["total"] += 1
        if r["_nseg"] >= VERY_LONG_MIN:
            cat_counts[cat]["vl"] += 1

    print(f"\n  [{label}]")
    print(f"  {'category':<20} {'total':>6}  {'16+':>5}  {'16+%':>7}")
    print("  " + "-" * 44)
    for cat in ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]:
        d = cat_counts.get(cat)
        if not d:
            continue
        pct = d["vl"] / d["total"] * 100
        print(f"  {cat:<20} {d['total']:>6}  {d['vl']:>5}  {pct:>6.1f}%")
    print(f"  {'합계':<20} {sum(d['total'] for d in cat_counts.values()):>6}")


# ── main ──────────────────────────────────────────────────────────────────────
def load_llm_rows(path: Path) -> list:
    """LLM 생성 전처리 완료 CSV 로드 및 _category/_nseg 메타 필드 추가."""
    if not path.exists():
        print(f"  [INFO] {path.name} 없음 — 건너뜀")
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            row["_category"] = (row.get("category") or "").strip()
            row["_nseg"]     = get_nseg(row)
            rows.append(row)
    print(f"  [INFO] {path.name}: {len(rows)}행 로드")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ratio", type=float, default=0.15,
                        help="일상 대화 내 Very Long 목표 비율 (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="저장 없이 통계만 출력")
    parser.add_argument("--no-llm-normal", action="store_true",
                        help="long_normal_preprocessed.csv 통합 건너뜀")
    parser.add_argument("--no-llm-phishing", action="store_true",
                        help="short_phishing_preprocessed.csv 통합 건너뜀")
    args = parser.parse_args()

    print("=" * 70)
    print("  Augmented Train 생성")
    print(f"  Z-score 임계값: {ZSCORE_THRESH}  |  목표 비율: {args.target_ratio*100:.0f}%  |  seed: {args.seed}")
    print("=" * 70)

    rows = load_rows(INPUT_CSV)
    print(f"\n원본 train: {len(rows)}행 로드")
    print_summary("원본 분포", rows)

    # 1. 아웃라이어 탐지
    outlier_ids = detect_outliers(rows)

    # 2. 아웃라이어 제거
    cleaned = [r for r in rows if (r.get("id") or r.get("\ufeffid") or "") not in outlier_ids]
    removed = len(rows) - len(cleaned)
    print(f"[제거 후] {len(rows)} → {len(cleaned)}행  (-{removed}개)")
    print_summary("아웃라이어 제거 후", cleaned)

    # 3. 일상 대화 concat 증강
    aug_rows = augment_normal(cleaned, args.target_ratio, args.seed)

    # 4. LLM 생성 증강 데이터 통합
    print("\n[LLM 증강 데이터 통합]")
    llm_normal_rows   = [] if args.no_llm_normal   else load_llm_rows(LLM_NORMAL_CSV)
    llm_phishing_rows = [] if args.no_llm_phishing else load_llm_rows(LLM_PHISHING_CSV)

    # 피싱 LLM 데이터: label=2/3 → category가 없으면 RISK_MAP 역매핑 사용
    for r in llm_phishing_rows:
        if r["_category"] not in PHISHING_CATS:
            lbl = str(r.get("label", "")).strip()
            r["_category"] = "대출 사기형" if lbl == "2" else "수사기관 사칭형"

    final = cleaned + aug_rows + llm_normal_rows + llm_phishing_rows
    print_summary("최종 (제거 + 증강 + LLM)", final)

    print(f"\n  [LLM 증강 추가 수]")
    print(f"  정상(long_normal) : {len(llm_normal_rows)}개")
    print(f"  피싱(short_phishing): {len(llm_phishing_rows)}개")

    # 5. 저장
    if args.dry_run:
        print("\n[dry-run] 파일 저장 생략")
    else:
        save_csv(final, OUTPUT_CSV)
        print(f"\n완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
