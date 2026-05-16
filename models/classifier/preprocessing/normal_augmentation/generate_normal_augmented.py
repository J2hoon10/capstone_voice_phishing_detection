"""
generate_normal_augmented.py

OpenAI Batch API를 사용하여 정상 대화(label=0) 데이터 700개 생성.
  - Track A (콜센터)  : minsang_train.csv에서 5개 샘플 → few-shot → 5개 생성
  - Track B (일상대화): normal_scripts.csv에서 5개 샘플  → few-shot → 5개 생성
  - 트랙별 호출 분리: 루프당 A 호출 1번 + B 호출 1번 = 2번
  - 70회 루프 × A×5 = 350개, 70회 루프 × B×5 = 350개 → 합계 700개

실행 방법:
  # 1단계: 배치 제출
  python generate_normal_augmented.py --mode submit

  # 2단계: 완료 확인
  python generate_normal_augmented.py --mode check

  # 3단계: 결과 수집
  python generate_normal_augmented.py --mode collect
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── 경로 ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / "transcriptions" / "normal_clean"
OUT_DIR    = BASE_DIR / "output"
PROMPT_DIR = BASE_DIR / "prompts"
OUT_DIR.mkdir(exist_ok=True)

NORMAL_CSV        = DATA_DIR / "normal_scripts.csv"
MINSANG_CSV       = DATA_DIR / "minsang_train.csv"
SYSTEM_PROMPT_TXT = PROMPT_DIR / "normal_system_prompt.txt"

BATCH_INPUT_JSONL = OUT_DIR / "normal_batch_input.jsonl"
BATCH_META_JSON   = OUT_DIR / "normal_batch_meta.json"
OUT_CSV           = OUT_DIR / "normal_augmented.csv"

# ── 설정 ──────────────────────────────────────────────────────────────────
MODEL   = "gpt-5.4-mini"
N_LOOPS = 70
SEED    = 42

# ── 시나리오 & 필수 키워드 ─────────────────────────────────────────────────
A_SCENARIOS = [
    "대출 심사 통과 후 정상적인 상환 방법과 이율을 안내하는 상황",
    "은행 전산 시스템 오류로 인해 고객의 납부 처리가 지연되고 있다고 양해를 구하는 상황",
    "비대면 계좌 개설 과정에서 필요한 개인정보 및 명의 확인 절차를 설명하는 상황",
    "고객의 카드 한도 조회 및 해외 거래 가능 여부를 상담해 주는 상황",
    "대출금 입금 요청이 접수되었으나 본인 확인 통화 절차가 필요하다고 안내하는 상황",
]
A_KEYWORDS = [
    "대출, 승인, 자금, 상환",
    "전산, 접수, 납부, 금액",
    "명의, 개설, 정보, 개인",
    "카드, 이용, 거래, 가능",
    "대출, 입금, 요청, 통화",
]

B_SCENARIOS = [
    "아침 뉴스에서 본 끔찍한 명의도용 사건을 지인 또는 가족에게 이야기하며 조심하라고 당부하는 상황",
    "모르는 번호로 온 수사관 사칭 전화를 방금 끊고 황당해하며 지인에게 말하는 상황",
    "최근 유행하는 불법 대포통장 개설 범죄에 관한 다큐멘터리 감상평을 나누는 상황",
    "서울에서 대규모 보이스피싱 일당(남성)이 검거되었다는 인터넷 기사를 단톡방에 공유하는 상황",
    "회사 보안 교육에서 배운 녹취 방법과 개인정보 보호의 중요성에 대해 동료와 푸념하는 상황",
]
B_KEYWORDS = [
    "사건, 수사, 명의, 도용",
    "수사관, 범죄, 연루, 조사",
    "불법, 대포, 개설, 피해자",
    "검거, 서울, 남성, 발견",
    "녹취, 진술, 압수, 정보",
]


# ── 텍스트 전처리 ──────────────────────────────────────────────────────────
def clean_script(text: str) -> str:
    """화자 레이블 제거 후 한 줄 연속 텍스트로 변환."""
    text = re.sub(r"^\d+\s*:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^(고객|상담사|손님|상담원)\s*:\s*", "", text, flags=re.MULTILINE)
    text = " ".join(text.split())
    return text.strip()


# ── 단일 트랙 프롬프트 구성 ────────────────────────────────────────────────
def build_single_track_prompt(
    track: str,
    shots: list[str],
    scenarios: list[str],
    keywords: list[str],
    n_per_scenario: int = 1,
) -> str:
    """트랙 하나만 생성하는 유저 프롬프트."""
    track_label = "A (콜센터)" if track == "A" else "B (일상대화)"
    n_total = len(scenarios) * n_per_scenario
    lines = []

    lines.append(f"[트랙 {track_label} 예시 대화]")
    for i, s in enumerate(shots, 1):
        lines.append(f"예시 {i}: {s[:400]}")

    lines.append("")
    lines.append("[생성 요청]")
    if n_per_scenario == 1:
        lines.append(f"아래 시나리오에 맞게 트랙 {track_label} 스크립트를 각 1개씩 생성하라.")
    else:
        lines.append(f"아래 시나리오에 맞게 트랙 {track_label} 스크립트를 각 {n_per_scenario}개씩 생성하라.")
        lines.append("같은 시나리오라도 서로 다른 내용과 표현 방식으로 작성하라.")
    lines.append("반드시 필수 키워드를 자연스럽게 포함하되, 범죄 맥락이 아닌 정상적인 맥락이어야 한다.")

    lines.append("")
    lines.append(f"[시나리오 목록]")
    for i, (scen, kw) in enumerate(zip(scenarios, keywords), 1):
        lines.append(f"{i}. 시나리오: {scen}")
        lines.append(f"   필수 키워드: {kw}")

    lines.append("")
    lines.append('【출력 규칙】')
    if n_per_scenario == 1:
        example_items = ", ".join([f'"{i}번스크립트"' for i in range(1, len(scenarios) + 1)])
        lines.append(f'- "scripts" 배열에 위 시나리오 순서대로 정확히 {len(scenarios)}개의 스크립트만 출력하라.')
    else:
        lines.append(f'- "scripts" 배열에 시나리오 순서대로 각 {n_per_scenario}개씩, 총 정확히 {n_total}개의 스크립트를 출력하라.')
        lines.append(f'- 배열 순서: 시나리오1의 1번째, 시나리오1의 2번째, 시나리오2의 1번째, 시나리오2의 2번째, ...')
        example_items = ", ".join([f'"{i}번스크립트"' for i in range(1, n_total + 1)])
    lines.append(f'- 반드시 JSON만 출력: {{"scripts": [{example_items}]}}')

    return "\n".join(lines)


# ── 배치 JSONL 생성 ────────────────────────────────────────────────────────
def build_batch_jsonl(
    system_prompt: str,
    normal_df: pd.DataFrame,
    minsang_df: pd.DataFrame,
    seed: int = SEED,
) -> None:
    import random
    rng = random.Random(seed)

    records = []
    for loop_idx in range(N_LOOPS):
        a_seed = rng.randint(0, 9999)
        b_seed = rng.randint(0, 9999)

        # 짝수 루프 → A 트랙 10개, 홀수 루프 → B 트랙 10개
        if loop_idx % 2 == 0:
            shots = [
                clean_script(str(row["script"]))
                for row in minsang_df.sample(5, random_state=a_seed).to_dict("records")
            ]
            track, scenarios, keywords = "A", A_SCENARIOS, A_KEYWORDS
        else:
            shots = [
                clean_script(str(row["script"]))
                for row in normal_df.sample(5, random_state=b_seed).to_dict("records")
            ]
            track, scenarios, keywords = "B", B_SCENARIOS, B_KEYWORDS

        records.append({
            "custom_id": f"loop_{loop_idx:04d}_{track}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": build_single_track_prompt(track, shots, scenarios, keywords, n_per_scenario=2)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 1.0,
                "max_completion_tokens": 8192,
            },
        })

    a_count = sum(1 for r in records if r["custom_id"].endswith("_A"))
    b_count = sum(1 for r in records if r["custom_id"].endswith("_B"))
    with open(BATCH_INPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[batch] JSONL 생성 완료: {BATCH_INPUT_JSONL} ({len(records)}개 요청, A={a_count}회×10개, B={b_count}회×10개)")


# ── 배치 제출 ──────────────────────────────────────────────────────────────
def submit_batch(client: OpenAI) -> str:
    print("[batch] 파일 업로드 중...")
    with open(BATCH_INPUT_JSONL, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    file_id = uploaded.id
    print(f"[batch] 파일 업로드 완료: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "normal_augmented_700"},
    )
    batch_id = batch.id
    print(f"[batch] 배치 제출 완료: {batch_id}")
    print(f"[batch] 상태: {batch.status}")

    meta = {"batch_id": batch_id, "file_id": file_id, "status": batch.status}
    BATCH_META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[batch] 메타 저장: {BATCH_META_JSON}")
    return batch_id


# ── 배치 취소 ──────────────────────────────────────────────────────────────
def cancel_batch(client: OpenAI, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    batch = client.batches.cancel(batch_id)
    print(f"[batch] 취소 요청 완료: {batch_id}")
    print(f"[batch] 상태: {batch.status}")


# ── 배치 상태 확인 ─────────────────────────────────────────────────────────
def check_batch(client: OpenAI, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"[batch] ID     : {batch_id}")
    print(f"[batch] 상태   : {batch.status}")
    print(f"[batch] 완료   : {counts.completed} / {counts.total}")
    print(f"[batch] 실패   : {counts.failed}")
    if batch.output_file_id:
        print(f"[batch] 출력파일: {batch.output_file_id}")


# ── 결과 수집 및 CSV 저장 ──────────────────────────────────────────────────
def collect_results(client: OpenAI, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    print("[collect] 배치 완료 대기 중...")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  상태: {batch.status} | {batch.request_counts.completed}/{batch.request_counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        print(f"[collect] 배치 실패: {batch.status}")
        return

    print(f"[collect] 결과 다운로드: {batch.output_file_id}")
    content = client.files.content(batch.output_file_id).content
    lines = content.decode("utf-8").strip().split("\n")

    rows = []
    uid = 1
    parse_errors = []

    for line in lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        # custom_id 형식: loop_XXXX_A 또는 loop_XXXX_B
        parts    = custom_id.split("_")
        loop_idx = int(parts[1])
        track    = parts[2]

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
            rows.append({
                "id":           f"normal_aug_{uid:04d}",
                "script":       str(script).strip(),
                "label":        0,
                "track":        track,
                "scenario_idx": scen_idx + 1,
                "loop_idx":     loop_idx,
            })
            uid += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[collect] 완료: {len(df)}개 저장 → {OUT_CSV}")
    if parse_errors:
        print(f"[collect] 파싱 실패 {len(parse_errors)}개: {parse_errors}")
    print(f"[collect] 트랙별:\n{df['track'].value_counts().to_string()}")


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["submit", "check", "collect", "cancel"],
        required=True,
        help="submit: 배치 생성·제출 | check: 상태 확인 | collect: 결과 수집 | cancel: 배치 취소",
    )
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()

    client = OpenAI()

    if args.mode == "submit":
        print("[load] CSV 로딩...")
        normal_df  = pd.read_csv(NORMAL_CSV)
        minsang_df = pd.read_csv(MINSANG_CSV)
        print(f"  normal_scripts : {len(normal_df)}행")
        print(f"  minsang_train  : {len(minsang_df)}행")

        system_prompt = SYSTEM_PROMPT_TXT.read_text(encoding="utf-8")
        build_batch_jsonl(system_prompt, normal_df, minsang_df)
        submit_batch(client)

    elif args.mode == "check":
        check_batch(client, args.batch_id)

    elif args.mode == "cancel":
        cancel_batch(client, args.batch_id)

    elif args.mode == "collect":
        collect_results(client, args.batch_id)


if __name__ == "__main__":
    main()
