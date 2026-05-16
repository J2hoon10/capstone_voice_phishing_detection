"""
sample_fewshot_sources.py

few-shot 원본 데이터(minsang_train.csv, normal_scripts.csv)에서
normal_augmented.csv의 트랙별 생성 개수만큼 무작위 추출하여
동일한 구조의 CSV를 생성한다.

  - Track A: minsang_train.csv  → 고객/상담사 화자 레이블 + 줄바꿈 제거
  - Track B: normal_scripts.csv → 숫자 화자 레이블(1 :, 2 :) + 줄바꿈 제거

실행:
  python sample_fewshot_sources.py
  python sample_fewshot_sources.py --seed 42
"""

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "transcriptions" / "normal_clean"
OUT_DIR    = BASE_DIR / "output"

NORMAL_CSV        = DATA_DIR / "normal_scripts.csv"
MINSANG_CSV       = DATA_DIR / "minsang_train.csv"
AUGMENTED_CSV     = OUT_DIR / "normal_augmented.csv"
OUT_CSV           = OUT_DIR / "normal_fewshot_sample.csv"

DEFAULT_SEED = 42


def clean_script(text: str) -> str:
    """화자 레이블 제거 후 한 줄 연속 텍스트로 변환."""
    # 숫자 화자 레이블 (1 : , 2 : 등)
    text = re.sub(r"^\d+\s*:\s*", "", text, flags=re.MULTILINE)
    # 한글 화자 레이블 (고객:, 상담사:, 손님:, 상담원: 등)
    text = re.sub(r"^(고객|상담사|손님|상담원)\s*:\s*", "", text, flags=re.MULTILINE)
    # 줄바꿈 포함 모든 공백을 단일 공백으로 합치기
    text = " ".join(text.split())
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # ── 생성 데이터 트랙별 개수 확인 ──────────────────────────────────────
    if not AUGMENTED_CSV.exists():
        print(f"[오류] {AUGMENTED_CSV} 파일이 없습니다. collect 먼저 실행하세요.")
        return

    aug_df = pd.read_csv(AUGMENTED_CSV)
    track_counts = aug_df["track"].value_counts()
    n_a = track_counts.get("A", 0)
    n_b = track_counts.get("B", 0)
    print(f"[info] normal_augmented.csv 트랙별 개수: A={n_a}, B={n_b}")

    # ── 원본 데이터 로딩 ──────────────────────────────────────────────────
    minsang_df = pd.read_csv(MINSANG_CSV)
    normal_df  = pd.read_csv(NORMAL_CSV)
    print(f"[info] minsang_train  : {len(minsang_df)}행")
    print(f"[info] normal_scripts : {len(normal_df)}행")

    if len(minsang_df) < n_a:
        print(f"[경고] minsang_train 행 수({len(minsang_df)})가 요청 수({n_a})보다 적습니다.")
        n_a = len(minsang_df)
    if len(normal_df) < n_b:
        print(f"[경고] normal_scripts 행 수({len(normal_df)})가 요청 수({n_b})보다 적습니다.")
        n_b = len(normal_df)

    # ── 무작위 샘플링 ─────────────────────────────────────────────────────
    a_sample = minsang_df.sample(n=n_a, random_state=args.seed).reset_index(drop=True)
    b_sample = normal_df.sample(n=n_b, random_state=args.seed).reset_index(drop=True)
    print(f"[info] 샘플링 완료: A={len(a_sample)}, B={len(b_sample)} (seed={args.seed})")

    # ── clean_script 적용 ─────────────────────────────────────────────────
    a_scripts = a_sample["script"].apply(lambda t: clean_script(str(t)))
    b_scripts = b_sample["script"].apply(lambda t: clean_script(str(t)))

    # 빈 문자열 확인
    a_empty = (a_scripts == "").sum()
    b_empty = (b_scripts == "").sum()
    if a_empty or b_empty:
        print(f"[경고] 빈 스크립트: A={a_empty}개, B={b_empty}개")

    # ── 결과 DataFrame 구성 ───────────────────────────────────────────────
    rows = []
    uid = 1

    for script in a_scripts:
        rows.append({
            "id":           f"normal_src_{uid:04d}",
            "script":       script,
            "label":        0,
            "track":        "A",
            "scenario_idx": 0,
            "loop_idx":     -1,
        })
        uid += 1

    for script in b_scripts:
        rows.append({
            "id":           f"normal_src_{uid:04d}",
            "script":       script,
            "label":        0,
            "track":        "B",
            "scenario_idx": 0,
            "loop_idx":     -1,
        })
        uid += 1

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[완료] {len(result_df)}개 저장 → {OUT_CSV}")
    print(f"[완료] 트랙별:\n{result_df['track'].value_counts().to_string()}")
    print(f"\n[샘플 확인] A 트랙 첫 번째:")
    print(f"  {result_df[result_df['track']=='A']['script'].iloc[0][:150]}...")
    print(f"\n[샘플 확인] B 트랙 첫 번째:")
    print(f"  {result_df[result_df['track']=='B']['script'].iloc[0][:150]}...")


if __name__ == "__main__":
    main()
