"""
retry_failed_loops.py

normal_augmented.csv에서 A!=5 또는 B!=5인 루프를 찾아
트랙 분리 호출(A-only / B-only)로 재배치 후 결과를 원본 CSV에 병합.

실행:
  python retry_failed_loops.py --mode submit
  python retry_failed_loops.py --mode check
  python retry_failed_loops.py --mode collect
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_normal_augmented import (
    MODEL, SEED, N_LOOPS,
    NORMAL_CSV, MINSANG_CSV, SYSTEM_PROMPT_TXT,
    A_SCENARIOS, A_KEYWORDS, B_SCENARIOS, B_KEYWORDS,
    clean_script, build_single_track_prompt,
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

ORIGINAL_CSV    = OUT_DIR / "normal_augmented.csv"
RETRY_JSONL     = OUT_DIR / "retry_batch_input.jsonl"
RETRY_META_JSON = OUT_DIR / "retry_batch_meta.json"


# ── 실패 루프 탐지 ─────────────────────────────────────────────────────────
def find_failed_loops(df: pd.DataFrame) -> list[int]:
    """짝수 루프(A 트랙 10개) 또는 홀수 루프(B 트랙 10개)가 맞지 않는 loop_idx 목록 반환."""
    counts = df.groupby(["loop_idx", "track"]).size().unstack(fill_value=0)
    failed = []
    for loop_idx in range(N_LOOPS):
        if loop_idx not in counts.index:
            failed.append(loop_idx)
            continue
        if loop_idx % 2 == 0:
            if counts.loc[loop_idx].get("A", 0) != 10:
                failed.append(loop_idx)
        else:
            if counts.loc[loop_idx].get("B", 0) != 10:
                failed.append(loop_idx)
    return sorted(failed)


# ── RNG 시드 재현 ──────────────────────────────────────────────────────────
def replay_rng_states(seed: int = SEED, n_loops: int = N_LOOPS) -> list[tuple[int, int]]:
    """원본 generate_normal_augmented.py와 동일한 RNG 시퀀스 재현."""
    import random
    rng = random.Random(seed)
    states = []
    for _ in range(n_loops):
        a_seed = rng.randint(0, 9999)
        b_seed = rng.randint(0, 9999)
        states.append((a_seed, b_seed))
    return states


# ── 재시도 배치 JSONL 생성 ─────────────────────────────────────────────────
def build_retry_jsonl(
    failed_loops: list[int],
    system_prompt: str,
    normal_df: pd.DataFrame,
    minsang_df: pd.DataFrame,
    existing_df: pd.DataFrame,
) -> None:
    rng_states = replay_rng_states()
    records = []

    for loop_idx in failed_loops:
        a_seed, b_seed = rng_states[loop_idx]

        # 짝수 루프 → A 재호출, 홀수 루프 → B 재호출
        if loop_idx % 2 == 0:
            shots = [
                clean_script(str(row["script"]))
                for row in minsang_df.sample(5, random_state=a_seed).to_dict("records")
            ]
            records.append({
                "custom_id": f"retry_loop_{loop_idx:04d}_A",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": build_single_track_prompt("A", shots, A_SCENARIOS, A_KEYWORDS, n_per_scenario=2)},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 1.0,
                    "max_completion_tokens": 8192,
                },
            })
        else:
            shots = [
                clean_script(str(row["script"]))
                for row in normal_df.sample(5, random_state=b_seed).to_dict("records")
            ]
            records.append({
                "custom_id": f"retry_loop_{loop_idx:04d}_B",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": build_single_track_prompt("B", shots, B_SCENARIOS, B_KEYWORDS, n_per_scenario=2)},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 1.0,
                    "max_completion_tokens": 8192,
                },
            })

    with open(RETRY_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[retry] JSONL 생성: {RETRY_JSONL} ({len(records)}개 요청)")
    print(f"[retry] 실패 루프: {failed_loops}")


# ── 배치 제출 ──────────────────────────────────────────────────────────────
def submit_batch(client: OpenAI) -> str:
    print("[retry] 파일 업로드 중...")
    with open(RETRY_JSONL, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    file_id = uploaded.id
    print(f"[retry] 파일 업로드 완료: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "normal_augmented_retry"},
    )
    batch_id = batch.id
    print(f"[retry] 배치 제출 완료: {batch_id}")
    print(f"[retry] 상태: {batch.status}")

    meta = {"batch_id": batch_id, "file_id": file_id, "status": batch.status}
    RETRY_META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return batch_id


# ── 배치 취소 ──────────────────────────────────────────────────────────────
def cancel_batch(client: OpenAI, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(RETRY_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    batch = client.batches.cancel(batch_id)
    print(f"[retry] 취소 요청 완료: {batch_id}")
    print(f"[retry] 상태: {batch.status}")


# ── 배치 상태 확인 ─────────────────────────────────────────────────────────
def check_batch(client: OpenAI, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(RETRY_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"[retry] ID     : {batch_id}")
    print(f"[retry] 상태   : {batch.status}")
    print(f"[retry] 완료   : {counts.completed} / {counts.total}")
    print(f"[retry] 실패   : {counts.failed}")
    if batch.output_file_id:
        print(f"[retry] 출력파일: {batch.output_file_id}")


# ── 결과 수집 및 원본 CSV 병합 ─────────────────────────────────────────────
def collect_and_merge(client: OpenAI, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(RETRY_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    print("[retry] 배치 완료 대기 중...")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  상태: {batch.status} | {batch.request_counts.completed}/{batch.request_counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        print(f"[retry] 배치 실패: {batch.status}")
        return

    content = client.files.content(batch.output_file_id).content
    lines = content.decode("utf-8").strip().split("\n")

    retry_rows = []
    parse_errors = []

    for line in lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        # custom_id 형식: retry_loop_XXXX_A 또는 retry_loop_XXXX_B
        parts    = custom_id.split("_")
        loop_idx = int(parts[2])
        track    = parts[3]

        if result.get("error"):
            print(f"  [오류] {custom_id}: {result['error']}")
            parse_errors.append(custom_id)
            continue

        raw_text = result["response"]["body"]["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except Exception:
                    print(f"  [파싱 실패] {custom_id}")
                    parse_errors.append(custom_id)
                    continue
            else:
                print(f"  [파싱 실패] {custom_id}")
                parse_errors.append(custom_id)
                continue

        scripts = parsed.get("scripts", [])
        if len(scripts) != 10:
            print(f"  [경고] {custom_id}: scripts={len(scripts)}개 (10개 기대)")

        for scen_idx, script in enumerate(scripts[:10]):
            retry_rows.append({
                "script":       str(script).strip(),
                "label":        0,
                "track":        track,
                "scenario_idx": scen_idx + 1,
                "loop_idx":     loop_idx,
            })

    if not retry_rows:
        print("[retry] 수집된 결과 없음.")
        return

    retry_df = pd.DataFrame(retry_rows)
    # 재시도로 교체되는 (loop_idx, track) 쌍
    replaced_pairs = set(zip(retry_df["loop_idx"], retry_df["track"]))

    # 원본 CSV에서 교체 대상 행 제거
    original_df = pd.read_csv(ORIGINAL_CSV)
    mask = original_df.apply(
        lambda r: (r["loop_idx"], r["track"]) in replaced_pairs, axis=1
    )
    original_df = original_df[~mask]

    # 병합 및 ID 재부여
    merged_df = pd.concat([original_df.drop(columns=["id"]), retry_df], ignore_index=True)
    merged_df = merged_df.sort_values(["loop_idx", "track", "scenario_idx"]).reset_index(drop=True)
    merged_df.insert(0, "id", [f"normal_aug_{i+1:04d}" for i in range(len(merged_df))])

    merged_df.to_csv(ORIGINAL_CSV, index=False, encoding="utf-8-sig")

    replaced_loops = sorted(set(r[0] for r in replaced_pairs))
    print(f"\n[retry] 병합 완료: 총 {len(merged_df)}건 → {ORIGINAL_CSV}")
    print(f"[retry] 교체된 루프: {replaced_loops}")
    if parse_errors:
        print(f"[retry] 파싱 실패: {parse_errors}")
    print(f"[retry] 트랙별:\n{merged_df['track'].value_counts().to_string()}")


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["submit", "check", "collect", "cancel"],
        required=True,
        help="submit: 실패 루프 탐지 및 재배치 제출 | check: 상태 확인 | collect: 결과 수집 및 병합 | cancel: 배치 취소",
    )
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()

    client = OpenAI()

    if args.mode == "submit":
        if not ORIGINAL_CSV.exists():
            print(f"[오류] {ORIGINAL_CSV} 파일이 없습니다.")
            return

        df = pd.read_csv(ORIGINAL_CSV)
        failed = find_failed_loops(df)

        if not failed:
            print("[retry] 실패 루프 없음. 모든 루프가 정상입니다.")
            return

        print(f"[retry] 실패 루프 {len(failed)}개 발견: {failed}")

        normal_df  = pd.read_csv(NORMAL_CSV)
        minsang_df = pd.read_csv(MINSANG_CSV)
        system_prompt = SYSTEM_PROMPT_TXT.read_text(encoding="utf-8")

        build_retry_jsonl(failed, system_prompt, normal_df, minsang_df, df)
        submit_batch(client)

    elif args.mode == "check":
        check_batch(client, args.batch_id)

    elif args.mode == "cancel":
        cancel_batch(client, args.batch_id)

    elif args.mode == "collect":
        collect_and_merge(client, args.batch_id)


if __name__ == "__main__":
    main()
