"""
build_minsang_csv.py

민원 상담 데이터의 train / val 폴더 내 JSON 파일들을 각각 CSV로 통합.

컬럼:
  - id            : 순번 (1-based)
  - file_name     : 원본 파일명
  - script        : consulting_content (전체 대화 내용)
  - 상위_폴더      : 하위 폴더명 (e.g. TS_하나카드)
  - category      : consulting_category (상담 카테고리)

출력:
  - preprocessing/minsang_train.csv
  - preprocessing/minsang_val.csv
"""

import json
from pathlib import Path
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR.parent / "data" / "normal" / "민원 상담"
OUT_TRAIN = BASE_DIR / "minsang_train.csv"
OUT_VAL   = BASE_DIR / "minsang_val.csv"


def build_df(split_dir: Path) -> pd.DataFrame:
    rows = []
    uid = 1

    for folder in sorted(split_dir.iterdir()):
        if not folder.is_dir():
            continue
        json_files = sorted(folder.glob("*.json"))
        print(f"  [{folder.name}] {len(json_files)}개 파일")

        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                data = json.loads(jf.read_text(encoding="cp949", errors="replace"))

            # JSON은 리스트 형태로 감싸져 있음
            if isinstance(data, list):
                data = data[0]

            rows.append({
                "id": uid,
                "file_name": jf.name,
                "script": data.get("consulting_content", ""),
                "상위_폴더": folder.name,
                "category": data.get("consulting_category", ""),
            })
            uid += 1

    return pd.DataFrame(rows, columns=["id", "file_name", "script", "상위_폴더", "category"])


print("[train]")
train_df = build_df(DATA_DIR / "train")
train_df.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
print(f"  → {OUT_TRAIN.name} ({len(train_df)}행)")
print("  카테고리 분포:")
for cat, cnt in train_df["category"].value_counts().items():
    print(f"    {cat}: {cnt}")

print("\n[val]")
val_df = build_df(DATA_DIR / "val")
val_df.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")
print(f"  → {OUT_VAL.name} ({len(val_df)}행)")
print("  카테고리 분포:")
for cat, cnt in val_df["category"].value_counts().items():
    print(f"    {cat}: {cnt}")
