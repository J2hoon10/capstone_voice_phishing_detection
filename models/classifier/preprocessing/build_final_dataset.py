"""
build_final_dataset.py

데이터셋 구성 규칙에 따라 train / val / test CSV를 생성한다.

[구성 규칙]
  - 피싱 원본 (noisy STT): 전체 사용
  - 일반 원본 (clean):     클래스별 N개 무작위 샘플링 (--normal-n-per-class)
  - 일반 증강:             원본 대비 1:R 비율 (--aug-ratio, default 4)
  - 피싱 증강:             클래스별 상한 지정 (--phishing-aug-per-class, default 무제한)

[데이터 소스]
  피싱 원본  : output/4class/train.csv  (source in ['original', 'asr_noise'])
  일반 원본  : output/4class/train.csv  (source == 'normal_asr')
  일반 증강  : normal_augmentation/output/normal_augmented.csv   (clean LLM)
               output/long_normal_augmented.csv                  (LLM + ASR noise)
  피싱 증강  : phishing_augmentation/output/phishing_augmented.csv
               output/short_phishing_augmented.csv

[사용법]
  python build_final_dataset.py --normal-n-per-class 150 --aug-ratio 4 --seed 42
  python build_final_dataset.py --dry-run          # 파일 저장 없이 통계만 확인
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "output" / "4class"
OUT_DIR   = BASE_DIR / "output" / "final_dataset"

TRAIN_CSV           = DATA_DIR / "train.csv"
NORMAL_AUG_CSV      = BASE_DIR / "normal_augmentation" / "output" / "normal_augmented.csv"
LONG_NORMAL_CSV     = BASE_DIR / "output" / "long_normal_augmented.csv"
PHISHING_AUG_CSV    = BASE_DIR / "phishing_augmentation" / "output" / "phishing_augmented.csv"
SHORT_PHISHING_CSV  = BASE_DIR / "output" / "short_phishing_augmented.csv"

# ── 상수 ──────────────────────────────────────────────────────────────────────
CAT_NORMAL   = {"상담 대화", "일상 대화"}
CAT_PHISHING = {"대출 사기형", "수사기관 사칭형"}
ALL_CATS     = ["상담 대화", "일상 대화", "대출 사기형", "수사기관 사칭형"]
LABEL_MAP    = {"상담 대화": "0", "일상 대화": "1", "대출 사기형": "2", "수사기관 사칭형": "3"}
TRACK_CAT    = {"A": "상담 대화", "B": "일상 대화"}

FIELDNAMES   = ["id", "text", "label", "category", "source", "filename", "segment_risks"]


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def _id(row: dict) -> str:
    """BOM이 있는 '\ufeffid' 컬럼도 처리."""
    return row.get("id") or row.get("\ufeffid") or ""


def normalize(row: dict, source_override: str = None) -> dict:
    """컬럼 스키마를 FIELDNAMES 기준으로 통일."""
    cat = row.get("category", "").strip()
    return {
        "id":            _id(row),
        "text":          (row.get("text") or row.get("script") or "").strip(),
        "label":         row.get("label", LABEL_MAP.get(cat, "")),
        "category":      cat,
        "source":        source_override or row.get("source", ""),
        "filename":      row.get("filename", ""),
        "segment_risks": row.get("segment_risks", ""),
    }


def group_by_cat(rows: list) -> dict:
    d = defaultdict(list)
    for r in rows:
        d[r["category"]].append(r)
    return d


def print_summary(label: str, rows: list):
    cats = Counter(r["category"] for r in rows)
    srcs = Counter(r["source"]   for r in rows)
    print(f"\n  [{label}]  총 {len(rows):,}행")
    for cat in ALL_CATS:
        if cat in cats:
            print(f"    {cat:<18}: {cats[cat]:>5}개")
    print(f"    source: {dict(srcs)}")


# ── 로더 ──────────────────────────────────────────────────────────────────────
def load_train_original(path: Path) -> tuple:
    """
    train.csv에서
      - 피싱 원본 (source in original/asr_noise)
      - 일반 원본 (source == normal_asr)
    을 분리해 반환.
    """
    phishing_orig, normal_orig = [], []
    with path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["id"] = _id(row)
            src = row.get("source", "").strip()
            cat = row.get("category", "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            nr = normalize(row)
            if src in ("original", "asr_noise") and cat in CAT_PHISHING:
                phishing_orig.append(nr)
            elif src == "normal_asr" and cat in CAT_NORMAL:
                normal_orig.append(nr)

    print(f"  피싱 원본 : {len(phishing_orig):>5}행  "
          f"({Counter(r['category'] for r in phishing_orig)})")
    print(f"  일반 원본 : {len(normal_orig):>5}행  "
          f"({Counter(r['category'] for r in normal_orig)})")
    return phishing_orig, normal_orig


def load_normal_aug_csv(path: Path) -> list:
    """
    normal_augmented.csv — 컬럼: id, script, label, track, scenario_idx, loop_idx
    track A → 상담 대화, B → 일상 대화
    """
    rows = []
    if not path.exists():
        print(f"  [없음] {path.name}")
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text  = (row.get("script") or row.get("text") or "").strip()
            track = row.get("track", "").strip()
            cat   = TRACK_CAT.get(track, "")
            if not text or not cat:
                continue
            rows.append({
                "id":            row.get("id", ""),
                "text":          text,
                "label":         LABEL_MAP[cat],
                "category":      cat,
                "source":        "llm_normal_aug_clean",
                "filename":      "",
                "segment_risks": "",
            })
    print(f"  [로드] {path.name}: {len(rows)}행  "
          f"({Counter(r['category'] for r in rows)})")
    return rows


def load_generic_csv(path: Path, source_override: str = None) -> list:
    """text 컬럼 기반 범용 로더."""
    rows = []
    if not path.exists():
        print(f"  [없음] {path.name}")
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            row["id"] = _id(row)
            rows.append(normalize(row, source_override))
    print(f"  [로드] {path.name}: {len(rows)}행  "
          f"({Counter(r['category'] for r in rows)})")
    return rows


# ── 분할 ──────────────────────────────────────────────────────────────────────
def stratified_split(rows_by_cat: dict, train_r: float, val_r: float,
                     seed: int) -> tuple:
    """카테고리별 비율을 유지하며 train / val / test 분할."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cat in ALL_CATS:
        rows = rows_by_cat.get(cat, [])
        if not rows:
            continue
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n        = len(shuffled)
        n_test   = max(1, round(n * (1 - train_r - val_r)))
        n_val    = max(1, round(n * val_r))
        n_train  = n - n_test - n_val

        test  += shuffled[:n_test]
        val   += shuffled[n_test:n_test + n_val]
        train += shuffled[n_test + n_val:n_test + n_val + n_train]
        print(f"    {cat:<18}: 전체 {n}  →  train {n_train} / val {n_val} / test {n_test}")
    return train, val, test


# ── 저장 ──────────────────────────────────────────────────────────────────────
def save_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [저장] {path}  ({len(rows):,}행)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="최종 학습 데이터셋 구성")
    parser.add_argument("--normal-n-per-class", type=int, default=150,
                        help="일반 원본 클래스당 샘플 수 (default: 150)")
    parser.add_argument("--aug-ratio", type=int, default=4,
                        help="일반 원본 대비 증강 배율 (default: 4 → 1:4)")
    parser.add_argument("--phishing-aug-per-class", type=int, default=0,
                        help="피싱 증강 클래스당 최대 수 (0=전체 사용, default: 0)")
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="학습 분할 비율 (default: 0.70)")
    parser.add_argument("--val-ratio",   type=float, default=0.15,
                        help="검증 분할 비율 (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="파일 저장 없이 통계만 출력")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 65)
    print("  최종 데이터셋 구성")
    print(f"  일반 원본 {args.normal_n_per_class}개/클래스  "
          f"| 증강 1:{args.aug_ratio}  "
          f"| seed={args.seed}")
    print("=" * 65)

    # ── 1. 원본 데이터 로드 ────────────────────────────────────────────────
    print("\n[1] 원본 데이터 로드")
    phishing_orig, normal_orig = load_train_original(TRAIN_CSV)

    # ── 2. 일반 원본 클래스별 무작위 샘플링 ───────────────────────────────
    n_target = args.normal_n_per_class
    normal_by_cat = group_by_cat(normal_orig)
    normal_sampled = []
    print(f"\n[2] 일반 원본 샘플링  ({n_target}개/클래스 목표)")
    for cat in sorted(CAT_NORMAL):
        pool = normal_by_cat.get(cat, [])
        n    = min(n_target, len(pool))
        if n < n_target:
            print(f"  {cat}: {n_target}개 요청, {len(pool)}개만 확보  ⚠")
        else:
            print(f"  {cat}: {len(pool)}개 중 {n}개 선택")
        normal_sampled.extend(rng.sample(pool, n))

    # ── 3. 일반 증강 로드 및 샘플링 ───────────────────────────────────────
    n_aug_target = n_target * args.aug_ratio
    print(f"\n[3] 일반 증강 로드  ({n_aug_target}개/클래스 목표, 1:{args.aug_ratio})")
    aug_pool = load_normal_aug_csv(NORMAL_AUG_CSV)
    aug_pool += load_generic_csv(LONG_NORMAL_CSV, source_override="llm_long_aug")

    aug_by_cat = group_by_cat(aug_pool)
    normal_aug_sampled = []
    for cat in sorted(CAT_NORMAL):
        pool = aug_by_cat.get(cat, [])
        n    = min(n_aug_target, len(pool))
        if n < n_aug_target:
            print(f"  {cat}: {n_aug_target}개 필요, {len(pool)}개만 확보  ⚠"
                  f"  (현재 증강 데이터 부족 — 생성기 재실행 후 재구성 권장)")
        else:
            print(f"  {cat}: {n_aug_target}개 선택 (풀 {len(pool)}개)")
        normal_aug_sampled.extend(rng.sample(pool, n))

    # ── 4. 피싱 증강 로드 ─────────────────────────────────────────────────
    print(f"\n[4] 피싱 증강 로드")
    phishing_aug = load_generic_csv(PHISHING_AUG_CSV, source_override="llm_phishing_aug")
    phishing_aug += load_generic_csv(SHORT_PHISHING_CSV)

    if args.phishing_aug_per_class > 0:
        ph_by_cat = group_by_cat(phishing_aug)
        phishing_aug = []
        for cat in CAT_PHISHING:
            pool = ph_by_cat.get(cat, [])
            n    = min(args.phishing_aug_per_class, len(pool))
            phishing_aug.extend(rng.sample(pool, n))
            print(f"  {cat}: {n}개 선택 (풀 {len(pool)}개)")
    else:
        print(f"  피싱 증강 전체 사용: {len(phishing_aug)}행")

    # ── 5. 전체 합산 ──────────────────────────────────────────────────────
    all_rows = phishing_orig + normal_sampled + normal_aug_sampled + phishing_aug
    print_summary("전체 합산", all_rows)

    # 일반 원본:증강 비율 확인
    print(f"\n  [일반 데이터 원본:증강 비율]")
    n_no = len(normal_sampled)
    n_na = len(normal_aug_sampled)
    print(f"    원본 {n_no}개 : 증강 {n_na}개  "
          f"= 1 : {n_na/n_no:.1f}" if n_no else "    원본 0개")

    # ── 6. 카테고리 유지 train/val/test 분할 ──────────────────────────────
    print(f"\n[6] 분할  (train {args.train_ratio:.0%} / val {args.val_ratio:.0%} / "
          f"test {1-args.train_ratio-args.val_ratio:.0%})")
    all_by_cat = group_by_cat(all_rows)
    train, val, test = stratified_split(
        all_by_cat, args.train_ratio, args.val_ratio, args.seed
    )
    print_summary("Train", train)
    print_summary("Val",   val)
    print_summary("Test",  test)

    # ── 7. 저장 ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] 파일 저장 생략")
    else:
        print(f"\n[7] 저장  →  {OUT_DIR}")
        save_csv(train, OUT_DIR / "train.csv")
        save_csv(val,   OUT_DIR / "val.csv")
        save_csv(test,  OUT_DIR / "test.csv")
        stats = {
            "config": vars(args),
            "total":  len(all_rows),
            "train":  len(train),
            "val":    len(val),
            "test":   len(test),
            "by_split": {
                "train": dict(Counter(r["category"] for r in train)),
                "val":   dict(Counter(r["category"] for r in val)),
                "test":  dict(Counter(r["category"] for r in test)),
            },
            "source_dist": dict(Counter(r["source"] for r in all_rows)),
        }
        (OUT_DIR / "dataset_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  [저장] dataset_stats.json")
        print(f"\n완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
