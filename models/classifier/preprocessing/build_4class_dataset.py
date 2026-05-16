"""
build_4class_dataset.py

4클래스 학습/검증/테스트 데이터셋 생성기

클래스 정의:
  0 - 상담 대화 (일반 Track A: 콜센터 상담 스타일 증강 대화)
  1 - 일상 대화 (일반 Track B: 일상 대화 스타일 증강 대화)
  2 - 대출 사기형 (보이스피싱 - 대출 빙자)
  3 - 수사기관 사칭형 (보이스피싱 - 수사기관/검찰 사칭, 바로 이 목소리 포함)

데이터 소스:
  - 일반(0, 1): normal_augmentation/output/normal_asr_noised.csv
               + normal_augmentation/output/normal_segment_risks.csv
  - 피싱(2, 3): phishing_augmentation/output/final/train+val+test.csv
               (label=1 행만 사용, label=0 콜센터 정상 데이터 제외)

분할: train 80% / val 10% / test 10%  (클래스별 stratified, seed=42)
출력: preprocessing/output/4class/{train,val,test}.csv + dataset_stats.json
컬럼: id, text, label, category, source, filename, segment_risks
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
    return re.sub(r" +", " ", cleaned).strip()

# ─── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
NORMAL_ASR_CSV   = BASE_DIR / "normal_augmentation"  / "output" / "normal_asr_noised.csv"
NORMAL_SEG_CSV   = BASE_DIR / "normal_augmentation"  / "output" / "normal_segment_risks.csv"
PHISHING_FINAL   = BASE_DIR / "phishing_augmentation" / "output" / "final"
OUTPUT_DIR       = BASE_DIR / "output" / "4class"

# ─── 상수 ────────────────────────────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED   = 42
CSV_COLUMNS  = ["id", "text", "label", "category", "source", "filename", "segment_risks"]

# 피싱 category → 새 label 매핑 (label=1 행만 대상)
PHISHING_CATEGORY_TO_LABEL = {
    "대출 사기형":   2,
    "수사기관 사칭형": 3,
    "바로 이 목소리": 3,
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
    """id → segment_risks 문자열 매핑"""
    risks: dict[str, str] = {}
    if not path.exists():
        return risks
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            risks[row["id"]] = row.get("segment_risks", "[]")
    return risks


def resolve_normal_seg_risks(asr_id: str, seg_map: dict[str, str]) -> str:
    """asr_normal_A_0001 → normal_A_0001 로 폴백 조회"""
    if asr_id in seg_map:
        return seg_map[asr_id]
    if asr_id.startswith("asr_"):
        return seg_map.get(asr_id[4:], "[]")
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

    # 최종 셔플 (클래스 순서 제거)
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


# ─── 데이터 로더 ──────────────────────────────────────────────────────────────
def load_normal_rows(asr_csv: Path, seg_csv: Path) -> list[dict]:
    asr_rows = load_csv(asr_csv)
    seg_map  = load_segment_risks(seg_csv)

    out = []
    for row in asr_rows:
        track = (row.get("track") or "").strip().upper()
        if track not in NORMAL_TRACK_MAP:
            continue
        label, category = NORMAL_TRACK_MAP[track]
        row_id  = (row.get("id") or "").strip()
        seg_str = resolve_normal_seg_risks(row_id, seg_map)
        out.append({
            "id":            row_id,
            "text":          clean_text((row.get("script") or "").strip()),
            "label":         label,
            "category":      category,
            "source":        "normal_asr",
            "filename":      "",
            "segment_risks": seg_str,
        })
    return out


def load_phishing_rows(final_dir: Path) -> list[dict]:
    """train/val/test 합산 후 label=1 행만 새 label(2/3)로 재매핑.

    asr_noise source는 제외: original + augmented_llm만 사용하여
    클래스당 ~500건을 유지 (asr_noise 포함 시 ~800건으로 중복 집계됨).
    """
    EXCLUDED_SOURCES = {"augmented_llm"}
    out = []
    for split_name in ("train", "val", "test"):
        csv_path = final_dir / f"{split_name}.csv"
        for row in load_csv(csv_path):
            if str(row.get("label", "")).strip() != "1":
                continue
            if (row.get("source") or "").strip() in EXCLUDED_SOURCES:
                continue
            category = (row.get("category") or "").strip()
            new_label = PHISHING_CATEGORY_TO_LABEL.get(category)
            if new_label is None:
                continue  # 알 수 없는 카테고리는 스킵
            out.append({
                "id":            row.get("id", ""),
                "text":          clean_text(row.get("text", "")),
                "label":         new_label,
                "category":      category,
                "source":        row.get("source", ""),
                "filename":      row.get("filename", ""),
                "segment_risks": row.get("segment_risks", "[]"),
            })
    return out


# ─── 통계 출력 ────────────────────────────────────────────────────────────────
def count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, ""))
        out[k] = out.get(k, 0) + 1
    return out


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="4클래스 학습 데이터셋 생성")
    parser.add_argument("--normal-asr",    default=str(NORMAL_ASR_CSV))
    parser.add_argument("--normal-seg",    default=str(NORMAL_SEG_CSV))
    parser.add_argument("--phishing-final",default=str(PHISHING_FINAL))
    parser.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--seed",  type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    print("[INFO] 일반 데이터 로드 중...")
    normal_rows  = load_normal_rows(Path(args.normal_asr), Path(args.normal_seg))
    print(f"  일반 행 수: {len(normal_rows)}")

    print("[INFO] 피싱 데이터 로드 중...")
    phishing_rows = load_phishing_rows(Path(args.phishing_final))
    print(f"  피싱 행 수: {len(phishing_rows)}")

    all_rows = normal_rows + phishing_rows
    print(f"[INFO] 전체 행 수: {len(all_rows)}")

    label_dist = count_by(all_rows, "label")
    print("[INFO] 클래스별 행 수:")
    label_names = {0: "상담 대화", 1: "일상 대화", 2: "대출 사기형", 3: "수사기관 사칭형"}
    for lbl in sorted(label_dist):
        print(f"  {lbl} ({label_names.get(int(lbl), '?')}): {label_dist[lbl]}")

    print(f"\n[INFO] {args.seed} 시드로 {SPLIT_RATIOS} 분할 중...")
    train_rows, val_rows, test_rows = stratified_split(all_rows, SPLIT_RATIOS, args.seed)

    out_dir = Path(args.output_dir)
    write_split(out_dir / "train.csv", train_rows)
    write_split(out_dir / "val.csv",   val_rows)
    write_split(out_dir / "test.csv",  test_rows)

    stats = {
        "total": len(all_rows),
        "splits": {
            "train": len(train_rows),
            "val":   len(val_rows),
            "test":  len(test_rows),
        },
        "class_distribution": {
            "total":  {str(k): v for k, v in label_dist.items()},
            "train":  count_by(train_rows, "label"),
            "val":    count_by(val_rows,   "label"),
            "test":   count_by(test_rows,  "label"),
        },
        "category_distribution": count_by(all_rows, "category"),
        "source_distribution":   count_by(all_rows, "source"),
        "label_names": label_names,
    }
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 출력 디렉터리: {out_dir}")
    print(f"  train : {out_dir / 'train.csv'}  ({len(train_rows)}건)")
    print(f"  val   : {out_dir / 'val.csv'}    ({len(val_rows)}건)")
    print(f"  test  : {out_dir / 'test.csv'}   ({len(test_rows)}건)")
    print(f"  stats : {stats_path}")

    print("\n[INFO] 분할별 클래스 분포:")
    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        dist = count_by(split_rows, "label")
        print(f"  {split_name}: { {k: dist.get(str(k), 0) for k in range(4)} }")


if __name__ == "__main__":
    main()
