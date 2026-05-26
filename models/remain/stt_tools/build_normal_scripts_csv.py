"""
build_normal_scripts_csv.py

일상 대화 데이터 폴더(VS_01~VS_05)의 모든 txt 파일을 하나의 CSV로 통합.

컬럼:
  - id        : 순번 (1-based)
  - file_name : 원본 파일명 (확장자 포함)
  - script    : 파일 전체 내용
  - 상위_폴더  : 원본 폴더명 (e.g. VS_01. KAKAO)

출력: preprocessing/normal_scripts.csv
"""

from pathlib import Path
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR.parent / "data" / "normal" / "일상 대화"
OUT_PATH  = BASE_DIR / "normal_scripts.csv"

FOLDERS = [
    "VS_01. KAKAO",
    "VS_02. FACEBOOK",
    "VS_03. INSTAGRAM",
    "VS_04. BAND",
    "VS_05. NATEON",
]

rows = []
uid = 1

for folder_name in FOLDERS:
    folder_path = DATA_DIR / folder_name
    if not folder_path.exists():
        print(f"[경고] 폴더 없음: {folder_path}")
        continue

    txt_files = sorted(folder_path.glob("*.txt"))
    print(f"[{folder_name}] {len(txt_files)}개 파일")

    for txt_path in txt_files:
        try:
            script = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            script = txt_path.read_text(encoding="cp949", errors="replace")

        rows.append({
            "id": uid,
            "file_name": txt_path.name,
            "script": script,
            "상위_폴더": folder_name,
        })
        uid += 1

df = pd.DataFrame(rows, columns=["id", "file_name", "script", "상위_폴더"])
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"\n[완료] 총 {len(df)}개 파일 → {OUT_PATH}")
print(df["상위_폴더"].value_counts().to_string())
