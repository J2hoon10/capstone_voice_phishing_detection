"""
build_4class_dataset.py

4클래스 학습/검증/테스트 데이터셋 생성기

클래스 정의:
  0 - 상담 대화 (일반 Track A)
  1 - 일상 대화 (일반 Track B)
  2 - 대출 사기형 (보이스피싱)
  3 - 수사기관 사칭형 (보이스피싱)

binary_label:
  0 - 일반 (label 0, 1)
  1 - 피싱 (label 2, 3)

데이터 구성 비율 (train/val/test 공통, 10/60/30):
  일반 (트랙당):
    - ASR 원본 10% (100건): normal_asr_noised.csv 중 원본 기반 (id가 asr_normal_aug_ 아닌 것)
    - ASR 증강 60% (600건): normal_asr_noised.csv 중 LLM 기반 (id가 asr_normal_aug_... 인 것)
    - 클린     30% (300건): normal_augmented.csv 중 ASR 증강에 미사용된 스크립트
  피싱 (카테고리당):
    - ASR 원본 10% (100건): asr_noised.csv 중 원본 전사 기반 (id가 asr_aug_ 아닌 것)
    - ASR 증강 60% (600건): asr_noised.csv 중 LLM 증강 기반 (id가 asr_aug_... 인 것)
    - 클린     30% (300건): phishing_augmented.csv 중 ASR 증강에 미사용된 스크립트

  → val/test 각 split은 70% ASR 노이즈 + 30% 클린 (이미지 계획: 60~70%/30~40% 충족)

데이터 소스:
  - 일반 noised : normal_augmentation/output/normal_asr_noised.csv
  - 일반 clean  : normal_augmentation/output/normal_augmented.csv
  - 일반 risks  : normal_augmentation/output/normal_segment_risks.csv
                  normal_augmentation/output/normal_aug_segment_risks.csv
  - 피싱 noised : phishing_augmentation/output/asr_noised.csv
  - 피싱 clean  : phishing_augmentation/output/phishing_augmented.csv

분할: train 60% / val 20% / test 20%  (클래스별 stratified, seed=42)
출력: preprocessing/output/4class/{train,val,test}.csv + dataset_stats.json
컬럼: id, text, label, binary_label, category, source, filename, segment_risks
"""

import argparse
import csv
import json
import os
import re
import random
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")


def clean_text(text: str) -> str:
    cleaned = _PUNCT_RE.sub("", text)
    cleaned = cleaned.replace("\n", " ")
    return re.sub(r" +", " ", cleaned).strip()


# ─── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
NORMAL_ASR_CSV     = BASE_DIR / "normal_augmentation" / "output" / "normal_asr_noised.csv"
NORMAL_AUG_CSV     = BASE_DIR / "normal_augmentation" / "output" / "normal_augmented.csv"
NORMAL_SEG_CSV     = BASE_DIR / "normal_augmentation" / "output" / "normal_segment_risks.csv"
NORMAL_AUG_SEG_CSV = BASE_DIR / "normal_augmentation" / "output" / "normal_aug_segment_risks.csv"
PHISHING_ASR_CSV   = BASE_DIR / "phishing_augmentation" / "output" / "asr_noised.csv"
PHISHING_LLM_CSV   = BASE_DIR / "phishing_augmentation" / "output" / "phishing_augmented.csv"
PHISHING_ORIG_CSV  = BASE_DIR / "transcriptions" / "gpu_small" / "phishing.csv"
PHISHING_SEG_CSV   = BASE_DIR / "stn_labeling" / "output" / "segment_risks.csv"
OUTPUT_DIR         = BASE_DIR / "output" / "4class"

# ─── 상수 ────────────────────────────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_SEED   = 42
CSV_COLUMNS  = ["id", "text", "label", "binary_label", "category", "source", "filename", "segment_risks"]

# 일반 데이터 구성 비율 (트랙당 고정 — 10/60/30)
NORMAL_ORIG_NOISED = 100   # ASR 원본  10%
NORMAL_LLM_NOISED  = 600   # ASR 증강  60%
NORMAL_LLM_CLEAN   = 300   # 클린      30%

# 피싱 데이터 구성 (카테고리별 동적)
# - 원본(phishing.csv): 전체 사용, 자연 노이즈 보유 → ASR 주입 불필요
# - LLM 증강 중 2/3: ASR 노이즈 적용 버전 (asr_noised.csv)
# - LLM 증강 중 1/3: 클린 그대로 (phishing_augmented.csv)
PHISHING_LLM_NOISED_RATIO = 2 / 3  # LLM 증강 중 ASR 노이즈 비율

# 피싱 category → label 매핑
PHISHING_CATEGORY_TO_LABEL = {
    "대출 사기형":    2,
    "수사기관 사칭형": 3,
}

# 일반 track → label / category 매핑
NORMAL_TRACK_MAP = {
    "A": (0, "상담 대화"),
    "B": (1, "일상 대화"),
}


# ─── 유틸 ────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_segment_risks(path: Path) -> dict[str, str]:
    """id → segment_risks 문자열 매핑. 파일이 없으면 빈 dict 반환."""
    risks: dict[str, str] = {}
    if not path.exists():
        return risks
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            risks[row["id"]] = row.get("segment_risks", "[]")
    return risks


def get_phishing_seg(row_id: str, seg_map: dict[str, str]) -> str:
    """피싱 row_id 로 segment_risks 조회.
    ASR 노이즈 행(asr_<base>_N)은 base ID로 폴백."""
    if row_id in seg_map:
        return seg_map[row_id]
    if row_id.startswith("asr_"):
        m = re.match(r"^asr_(.+)_\d+$", row_id)
        if m:
            base = m.group(1)
            if base in seg_map:
                return seg_map[base]
    return "[]"


def stratified_split(rows: list[dict], ratios: dict, seed: int) -> tuple[list, list, list]:
    """클래스(label)별로 동일 비율 분할 후 합산."""
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        k = str(row["label"])
        by_label.setdefault(k, []).append(row)

    rng = random.Random(seed)
    train_all, val_all, test_all = [], [], []

    for label, label_rows in sorted(by_label.items()):
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios["train"])
        n_val   = int(n * ratios["val"])
        train_all.extend(shuffled[:n_train])
        val_all.extend(shuffled[n_train:n_train + n_val])
        test_all.extend(shuffled[n_train + n_val:])

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)
    return train_all, val_all, test_all


def write_split(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ─── 일반 데이터 로더 ──────────────────────────────────────────────────────────
def load_normal_rows(
    asr_csv: Path,
    aug_csv: Path,
    seg_csv: Path,
    aug_seg_csv: Path,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    # segment_risks 맵: 원본 기반 + LLM 기반 둘 다 로드
    seg_map     = load_segment_risks(seg_csv)
    aug_seg_map = load_segment_risks(aug_seg_csv)

    def get_seg(row_id: str) -> str:
        """ID 직접 조회 → asr_ 제거 후 폴백 조회."""
        if row_id in seg_map:
            return seg_map[row_id]
        if row_id in aug_seg_map:
            return aug_seg_map[row_id]
        stripped = row_id[4:] if row_id.startswith("asr_") else row_id
        return seg_map.get(stripped) or aug_seg_map.get(stripped, "[]")

    # ── normal_asr_noised.csv: 원본 노이즈 / LLM 노이즈 분리 ─────────────────
    asr_rows = load_csv(asr_csv)
    by_track_orig_noised: dict[str, list[dict]] = {}
    by_track_llm_noised:  dict[str, list[dict]] = {}

    for row in asr_rows:
        track = (row.get("track") or "").strip().upper()
        if track not in NORMAL_TRACK_MAP:
            continue
        if row.get("id", "").startswith("asr_normal_aug_"):
            by_track_llm_noised.setdefault(track, []).append(row)
        else:
            by_track_orig_noised.setdefault(track, []).append(row)

    # ── normal_augmented.csv: 클린 LLM ───────────────────────────────────────
    aug_rows = load_csv(aug_csv)
    by_track_llm_clean: dict[str, list[dict]] = {}
    for row in aug_rows:
        track = (row.get("track") or "").strip().upper()
        if track in NORMAL_TRACK_MAP:
            by_track_llm_clean.setdefault(track, []).append(row)

    # ── 구성 비율 샘플링 ─────────────────────────────────────────────────────
    out = []
    for track, (label, category) in NORMAL_TRACK_MAP.items():
        orig_pool      = list(by_track_orig_noised.get(track, []))
        llm_noised_pool = list(by_track_llm_noised.get(track, []))
        llm_clean_pool  = list(by_track_llm_clean.get(track, []))

        rng.shuffle(orig_pool)
        rng.shuffle(llm_noised_pool)

        # LLM 노이즈 600건 선택 후, 그 기반 ID는 클린 풀에서 제외 (중복 방지)
        selected_llm_noised = llm_noised_pool[:NORMAL_LLM_NOISED]
        used_base_ids = {r["id"][4:] for r in selected_llm_noised}  # "asr_" 제거
        clean_filtered = [r for r in llm_clean_pool if r["id"] not in used_base_ids]
        rng.shuffle(clean_filtered)

        selected_orig  = orig_pool[:NORMAL_ORIG_NOISED]
        selected_clean = clean_filtered[:NORMAL_LLM_CLEAN]

        print(
            f"  [Track {track}] 원본노이즈={len(selected_orig)}, "
            f"LLM노이즈={len(selected_llm_noised)}, "
            f"클린={len(selected_clean)}"
        )

        for group, source_tag in [
            (selected_orig,        "normal_asr_orig"),
            (selected_llm_noised,  "normal_asr_llm"),
            (selected_clean,       "normal_clean_llm"),
        ]:
            for row in group:
                row_id = (row.get("id") or "").strip()
                out.append({
                    "id":            row_id,
                    "text":          clean_text((row.get("script") or "").strip()),
                    "label":         label,
                    "binary_label":  0,
                    "category":      category,
                    "source":        source_tag,
                    "filename":      "",
                    "segment_risks": get_seg(row_id),
                })
    return out


# ─── 피싱 데이터 로더 ──────────────────────────────────────────────────────────
def load_phishing_rows(
    asr_csv: Path,
    llm_csv: Path,
    orig_csv: Path,
    seed: int,
    max_orig_len: int | None = None,
    phishing_seg_csv: Path | None = None,
) -> list[dict]:
    """카테고리별 구성:
      - phishing_original (전체): phishing.csv 원본 전사 — 자연 노이즈 보유, ASR 주입 불필요
      - phishing_asr_llm  (2/3):  asr_noised.csv 중 LLM 증강 기반 (id: asr_aug_*)
      - phishing_clean_llm(1/3):  phishing_augmented.csv 중 노이즈 미사용 행

    카테고리 균형 전략:
      target_total = min(n_orig + n_llm_avail) across all categories
      각 카테고리: n_llm_budget = target_total - n_orig
                   n_llm_noised = int(n_llm_budget * 2/3), n_llm_clean = n_llm_budget - n_llm_noised
    → 원본 전량 유지 + LLM 예산 조정으로 카테고리별 총량을 균등하게 맞춤
    """
    rng = random.Random(seed)

    asr_rows_all  = load_csv(asr_csv)
    llm_clean_all = load_csv(llm_csv)
    orig_all      = load_csv(orig_csv)

    # 피싱 segment_risks 맵 로드
    phishing_seg_map = load_segment_risks(phishing_seg_csv) if phishing_seg_csv else {}
    if phishing_seg_map:
        print(f"  [segment_risks] 피싱 위험도 로드: {len(phishing_seg_map)}건")
    else:
        print("  [segment_risks] 피싱 위험도 없음 (--phishing-seg 미지정 또는 파일 없음) → 전부 []")

    # ── 1차 패스: 카테고리별 가용량 파악 ─────────────────────────────────────
    cat_pools: dict[str, dict] = {}
    for category, label in sorted(PHISHING_CATEGORY_TO_LABEL.items()):
        cat_orig_all = [r for r in orig_all
                        if (r.get("category") or "").strip() == category]
        if max_orig_len is not None:
            before = len(cat_orig_all)
            cat_orig = [r for r in cat_orig_all if len((r.get("text") or "").strip()) <= max_orig_len]
            filtered = before - len(cat_orig)
            if filtered:
                print(f"  [필터] {category}: {before}건 → {len(cat_orig)}건 ({filtered}건 제거, >{max_orig_len}자)")
        else:
            cat_orig = cat_orig_all
        cat_asr_llm = [r for r in asr_rows_all
                       if (r.get("category") or "").strip() == category
                       and r.get("id", "").startswith("asr_aug_")]
        cat_pools[category] = {
            "label":    label,
            "orig":     cat_orig,
            "asr_llm":  cat_asr_llm,
        }

    # ── target_total = min(n_orig + n_llm_avail) ─────────────────────────────
    target_total = min(
        len(p["orig"]) + len(p["asr_llm"])
        for p in cat_pools.values()
    )

    # ── 2차 패스: target_total 기준으로 LLM 예산 조정 후 샘플링 ──────────────
    out = []
    for category, pool in sorted(cat_pools.items()):
        label       = pool["label"]
        cat_orig    = pool["orig"]
        cat_asr_llm = list(pool["asr_llm"])
        rng.shuffle(cat_asr_llm)

        n_llm_budget = target_total - len(cat_orig)
        n_llm_noised = int(n_llm_budget * PHISHING_LLM_NOISED_RATIO)
        n_llm_clean  = n_llm_budget - n_llm_noised

        sel_llm_noised = cat_asr_llm[:n_llm_noised]

        # LLM 클린: 노이즈에 사용된 소스 행 제외
        used_source_ids = {
            r["id"][4:].rsplit("_", 1)[0]
            for r in sel_llm_noised
            if len(r.get("id", "")) > 4
        }
        clean_pool = [r for r in llm_clean_all
                      if (r.get("category") or "").strip() == category
                      and r.get("id", "") not in used_source_ids]
        rng.shuffle(clean_pool)
        sel_llm_clean = clean_pool[:n_llm_clean]

        print(
            f"  [피싱 {category}] "
            f"원본={len(cat_orig)}, LLM노이즈={len(sel_llm_noised)}, LLM클린={len(sel_llm_clean)}"
            f"  (합계={len(cat_orig)+len(sel_llm_noised)+len(sel_llm_clean)}, target={target_total})"
        )

        for group, source_tag in [
            (cat_orig,         "phishing_original"),
            (sel_llm_noised,   "phishing_asr_llm"),
            (sel_llm_clean,    "phishing_clean_llm"),
        ]:
            for row in group:
                row_id = (row.get("id") or "").strip()
                seg = (get_phishing_seg(row_id, phishing_seg_map)
                       if phishing_seg_map else row.get("segment_risks", "[]"))
                out.append({
                    "id":            row_id,
                    "text":          clean_text((row.get("text") or "").strip()),
                    "label":         label,
                    "binary_label":  1,
                    "category":      category,
                    "source":        source_tag,
                    "filename":      row.get("filename", ""),
                    "segment_risks": seg,
                })

    return out


# ─── 통계 ────────────────────────────────────────────────────────────────────
def count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, ""))
        out[k] = out.get(k, 0) + 1
    return out


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="4클래스 학습 데이터셋 생성")
    parser.add_argument("--normal-asr",      default=str(NORMAL_ASR_CSV))
    parser.add_argument("--normal-aug",      default=str(NORMAL_AUG_CSV))
    parser.add_argument("--normal-seg",      default=str(NORMAL_SEG_CSV))
    parser.add_argument("--normal-aug-seg",  default=str(NORMAL_AUG_SEG_CSV))
    parser.add_argument("--phishing-asr",  default=str(PHISHING_ASR_CSV))
    parser.add_argument("--phishing-llm",  default=str(PHISHING_LLM_CSV))
    parser.add_argument("--phishing-orig", default=str(PHISHING_ORIG_CSV))
    parser.add_argument(
        "--phishing-seg", default=str(PHISHING_SEG_CSV),
        help="피싱 segment_risks CSV (stn_labeling/output/segment_risks.csv)",
    )
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--max-orig-len", type=int, default=2800,
        help="피싱 원본 데이터 최대 텍스트 길이 (초과 행 제외). 기본값: 2800",
    )
    args = parser.parse_args()

    print("[INFO] 일반 데이터 로드 중...")
    normal_rows = load_normal_rows(
        Path(args.normal_asr),
        Path(args.normal_aug),
        Path(args.normal_seg),
        Path(args.normal_aug_seg),
        args.seed,
    )
    print(f"  일반 행 수: {len(normal_rows)}")

    print("[INFO] 피싱 데이터 로드 중...")
    phishing_rows = load_phishing_rows(
        Path(args.phishing_asr),
        Path(args.phishing_llm),
        Path(args.phishing_orig),
        args.seed,
        max_orig_len=args.max_orig_len,
        phishing_seg_csv=Path(args.phishing_seg) if args.phishing_seg else None,
    )
    print(f"  피싱 행 수: {len(phishing_rows)}")

    all_rows = normal_rows + phishing_rows
    print(f"[INFO] 전체 행 수: {len(all_rows)}")

    label_names = {0: "상담 대화", 1: "일상 대화", 2: "대출 사기형", 3: "수사기관 사칭형"}
    label_dist  = count_by(all_rows, "label")
    print("[INFO] 클래스별 행 수:")
    for lbl in sorted(label_dist):
        print(f"  {lbl} ({label_names.get(int(lbl), '?')}): {label_dist[lbl]}")

    print(f"\n[INFO] seed={args.seed}, 분할={SPLIT_RATIOS}")
    train_rows, val_rows, test_rows = stratified_split(all_rows, SPLIT_RATIOS, args.seed)

    out_dir = Path(args.output_dir)
    write_split(out_dir / "train.csv", train_rows)
    write_split(out_dir / "val.csv",   val_rows)
    write_split(out_dir / "test.csv",  test_rows)

    stats = {
        "total":  len(all_rows),
        "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "class_distribution": {
            "total": {str(k): v for k, v in label_dist.items()},
            "train": count_by(train_rows, "label"),
            "val":   count_by(val_rows,   "label"),
            "test":  count_by(test_rows,  "label"),
        },
        "source_distribution":   count_by(all_rows, "source"),
        "category_distribution": count_by(all_rows, "category"),
        "label_names": label_names,
    }
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 출력: {out_dir}")
    print(f"  train : {len(train_rows)}건")
    print(f"  val   : {len(val_rows)}건")
    print(f"  test  : {len(test_rows)}건")

    print("\n[INFO] 분할별 클래스 분포:")
    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        dist = count_by(split_rows, "label")
        print(f"  {split_name}: { {k: dist.get(str(k), 0) for k in range(4)} }")


if __name__ == "__main__":
    main()
