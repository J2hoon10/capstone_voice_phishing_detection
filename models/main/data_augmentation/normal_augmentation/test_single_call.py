"""
test_single_call.py

generate_normal_augmented.py의 프롬프트 로직을 단건 호출로 테스트.
loop_idx=0 기준 샘플 1회 실행 후 결과 출력.
"""

import json
import random
from pathlib import Path

import pandas as pd
from openai import OpenAI

OUT_PATH = Path(__file__).resolve().parent / "output" / "test_single_call.csv"

# generate_normal_augmented.py와 동일한 설정 import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_normal_augmented import (
    MODEL, SEED,
    NORMAL_CSV, MINSANG_CSV, SYSTEM_PROMPT_TXT,
    A_SCENARIOS, A_KEYWORDS, B_SCENARIOS, B_KEYWORDS,
    clean_script, build_single_track_prompt,
)

def run_track_test(
    track: str,
    shots: list,
    scenarios: list,
    keywords: list,
    system_prompt: str,
    client,
) -> list[dict]:
    user_prompt = build_single_track_prompt(track, shots, scenarios, keywords, n_per_scenario=2)
    print(f"[test] 트랙 {track} 유저 프롬프트 길이: {len(user_prompt)}자")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=1.0,
        max_completion_tokens=8192,
    )

    raw = response.choices[0].message.content
    usage = response.usage
    print(f"[test] 트랙 {track} 토큰: 입력={usage.prompt_tokens} / 출력={usage.completion_tokens} / 합계={usage.total_tokens}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[test] 트랙 {track} JSON 파싱 실패. 원본:\n{raw}")
        return []

    scripts = parsed.get("scripts", [])
    n_expected = len(scenarios) * 2
    if len(scripts) != n_expected:
        print(f"[test] 트랙 {track} 경고: scripts={len(scripts)}개 ({n_expected}개 기대)")
    rows = []
    print(f"\n{'='*20} 트랙 {track} {'='*20}")
    for i, script in enumerate(scripts, 1):
        scen_idx = (i - 1) // 2  # 시나리오 인덱스 (0-based)
        variant   = (i - 1) % 2 + 1  # 변형 번호 (1 또는 2)
        scen = scenarios[scen_idx] if scen_idx < len(scenarios) else "(없음)"
        kw   = keywords[scen_idx]  if scen_idx < len(keywords)  else "(없음)"
        print(f"\n[{track}{i}] 시나리오 {scen_idx+1}-변형{variant}: {scen}")
        print(f"  길이: {len(script)}자")
        print(f"  내용: {script[:200]}{'...' if len(script) > 200 else ''}")
        rows.append({
            "id":           f"test_{track}{i}",
            "track":        track,
            "scenario_idx": i,
            "scenario":     scen,
            "keywords":     kw,
            "script":       script,
            "label":        0,
        })
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="단건 호출 테스트")
    parser.add_argument(
        "--loop-idx", type=int, default=0,
        help="재현할 loop_idx (짝수=A, 홀수=B). 기본값=0",
    )
    args = parser.parse_args()
    loop_idx = args.loop_idx

    print(f"[test] 모델: {MODEL}")
    print(f"[test] loop_idx={loop_idx}  →  트랙 {'A' if loop_idx % 2 == 0 else 'B'}")

    normal_df  = pd.read_csv(NORMAL_CSV)
    minsang_df = pd.read_csv(MINSANG_CSV)
    system_prompt = SYSTEM_PROMPT_TXT.read_text(encoding="utf-8")
    print(f"[test] 시스템 프롬프트 길이: {len(system_prompt)}자")
    print("-" * 60)

    # loop_idx까지 RNG 시퀀스 재현 (generate_normal_augmented.py와 동일)
    rng = random.Random(SEED)
    a_seed = b_seed = 0
    for _ in range(loop_idx + 1):
        a_seed = rng.randint(0, 9999)
        b_seed = rng.randint(0, 9999)

    client = OpenAI()
    rows = []
    if loop_idx % 2 == 0:  # A 트랙
        shots = [
            clean_script(str(row["script"]))
            for row in minsang_df.sample(5, random_state=a_seed).to_dict("records")
        ]
        rows += run_track_test("A", shots, A_SCENARIOS, A_KEYWORDS, system_prompt, client)
    else:  # B 트랙
        shots = [
            clean_script(str(row["script"]))
            for row in normal_df.sample(5, random_state=b_seed).to_dict("records")
        ]
        rows += run_track_test("B", shots, B_SCENARIOS, B_KEYWORDS, system_prompt, client)

    OUT_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[test] 결과 저장: {OUT_PATH}")
    print("[test] 완료")

if __name__ == "__main__":
    main()
