"""
generate_normal_augmented.py

OpenAI Batch API를 사용하여 정상 대화(label=0) 데이터 1800개 생성.
  - Track A (콜센터)  : minsang_train.csv에서 5개 샘플 → few-shot → 2개 생성
  - Track B (일상대화): normal_scripts.csv에서 5개 샘플  → few-shot → 2개 생성
  - 트랙별 호출 분리: 루프당 A 또는 B 중 하나, 2개 시나리오 순환
  - 900회 루프 × (A 450회×2 + B 450회×2) → 합계 1800개 (900/track)

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

BATCH_INPUT_JSONL       = OUT_DIR / "normal_batch_input.jsonl"
BATCH_META_JSON         = OUT_DIR / "normal_batch_meta.json"
RETRY_BATCH_INPUT_JSONL = OUT_DIR / "normal_retry_batch_input.jsonl"
RETRY_BATCH_META_JSON   = OUT_DIR / "normal_retry_batch_meta.json"
OUT_CSV                 = OUT_DIR / "normal_augmented.csv"

# ── 설정 ──────────────────────────────────────────────────────────────────
MODEL              = "gpt-5.4-mini"
N_LOOPS            = 900   # 450 A + 450 B = 900 루프 × 2개/루프 = 1800개 (900/track)
SCENARIOS_PER_CALL = 2     # 호출당 시나리오(=스크립트) 수
SEED               = 42

# ── 시나리오 & 필수 키워드 ─────────────────────────────────────────────────
A_SCENARIOS = [
    "은행 대출상담팀 과장이 대출 심사 통과 후 상환 방법과 이율을 안내하는 상황",
    "은행 고객센터 상담원이 전산 오류로 납부 처리가 지연된다고 양해를 구하는 상황",
    "은행 대리가 비대면 계좌 개설 시 개인정보 및 명의 확인 절차를 설명하는 상황",
    "카드사 상담팀 차장이 카드 한도 조회 및 해외 거래 가능 여부를 안내하는 상황",
    "은행 여신팀 팀장이 대출금 실행 전 본인 확인 통화 절차를 안내하는 상황",
    "고객이 경찰청 민원콜센터에 전화해 보이스피싱 피해 사실을 신고하고 담당자에게 접수 절차를 안내받는 상황",
    "고객이 금융감독원 소비자보호센터에 전화해 불법 금융상품 피해를 상담하는 상황",
    "고객이 사이버수사대에 온라인 사기 피해를 신고하고 담당 수사관에게 사건 접수 안내를 받는 상황",
    # 설명형: 한 화자가 길게 설명하는 시나리오
    "은행 여신팀 직원이 중도상환 수수료 계산 방식부터 대환 조건·금리 비교까지 고객에게 단계별로 상세히 설명하는 상황",
    "카드사 고객보호팀 직원이 부정 거래 이의신청 절차·피해 금액 회수 방법·계정 보호 조치를 단계별로 상세히 안내하는 상황",
]
A_KEYWORDS = [
    "과장, 대출, 승인, 상환",
    "상담원, 납부, 전산, 금액",
    "대리, 명의, 계좌, 개설",
    "차장, 카드, 한도, 거래",
    "팀장, 대출, 실행, 입금",
    "경찰청, 신고, 피해, 접수",
    "금융감독원, 민원, 상담, 피해",
    "사이버수사대, 수사관, 접수, 사건",
    "중도상환, 수수료, 대환, 금리",
    "부정거래, 이의신청, 피해, 보호",
]

B_SCENARIOS = [
    "아침 뉴스에서 본 명의도용 사건을 지인에게 이야기하며 계좌 동결·압수 피해 사례를 구체적으로 설명하고 조심하라고 당부하는 상황",
    "모르는 번호로 온 검사 사칭 전화를 방금 끊고 황당해하며 지인에게 통화 내용을 상세히 재현해서 말하는 상황",
    "가족 중 한 명이 대출 사기 전화를 받고 송금 직전에 멈춘 아찔한 경험을 친척들에게 자세히 설명하는 상황",
    "직장 동료가 신용점수 올려준다는 대환대출 전화에 속을 뻔했던 경험을 점심시간에 생생하게 이야기하는 상황",
    "부모님께 보이스피싱 수법을 알려드리며 계좌번호·비밀번호를 절대 알려주면 안 된다고 구체적인 사례를 들어 설명하는 상황",
    "친구가 금융감독원 사칭 전화를 받아 계좌 동결 협박을 당한 경험을 듣고 함께 대처 방법을 이야기하는 상황",
    "회사 보안 교육에서 배운 개인정보 도용·압수 수법을 동료와 복기하며 실제 사례와 예방법을 이야기하는 상황",
    # 설명형: 한 화자가 길게 설명하는 시나리오
    "보이스피싱으로 수백만 원을 잃은 지인의 피해 경위를 처음 듣는 가족에게 처음부터 끝까지 순서대로 재현해 설명하는 상황",
    "대출 신청이 처음인 친구에게 심사 과정·필요 서류·금리 계산 방법을 단계별로 차근차근 설명해주는 상황",
    "스마트폰이 익숙하지 않은 부모님께 모바일 뱅킹 앱 설치부터 계좌 이체·보안 주의사항까지 전 과정을 설명하는 상황",
]
B_KEYWORDS = [
    "명의도용, 계좌, 동결, 압수",
    "검사, 사칭, 수사, 연루",
    "대출, 송금, 사기, 피해",
    "신용점수, 대환, 이체, 한도",
    "계좌번호, 비밀번호, 개인정보, 보호",
    "금융감독원, 동결, 협박, 대처",
    "도용, 압수, 개인정보, 예방",
    "피해, 이체, 계좌, 경위",
    "대출, 심사, 서류, 금리",
    "이체, 계좌번호, 보안, 비밀번호",
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

        # 짝수 루프 → A 트랙, 홀수 루프 → B 트랙
        # 트랙 내 루프 순번(track_loop_idx)을 기준으로 시나리오 2개씩 순환
        track_loop_idx = loop_idx // 2
        scen_offset = (track_loop_idx * SCENARIOS_PER_CALL) % 10  # 10개 시나리오 순환

        if loop_idx % 2 == 0:
            shots = [
                clean_script(str(row["script"]))
                for row in minsang_df.sample(5, random_state=a_seed).to_dict("records")
            ]
            track      = "A"
            all_scen   = A_SCENARIOS
            all_kw     = A_KEYWORDS
        else:
            shots = [
                clean_script(str(row["script"]))
                for row in normal_df.sample(5, random_state=b_seed).to_dict("records")
            ]
            track      = "B"
            all_scen   = B_SCENARIOS
            all_kw     = B_KEYWORDS

        # SCENARIOS_PER_CALL=2개 시나리오 순환 선택 (인덱스 범위 초과 시 wraparound)
        indices  = [scen_offset % 10, (scen_offset + 1) % 10]
        scenarios = [all_scen[i] for i in indices]
        keywords  = [all_kw[i]   for i in indices]

        records.append({
            "custom_id": f"loop_{loop_idx:04d}_{track}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": build_single_track_prompt(track, shots, scenarios, keywords, n_per_scenario=1)},
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

    print(f"[batch] JSONL 생성 완료: {BATCH_INPUT_JSONL} ({len(records)}개 요청, A={a_count}회×{SCENARIOS_PER_CALL}개, B={b_count}회×{SCENARIOS_PER_CALL}개)")


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


# ── 재시도: 실패/부족 루프 탐색 ───────────────────────────────────────────
def find_missing_loops() -> list[int]:
    """normal_augmented.csv에서 scripts < 2인 loop_idx 목록 반환."""
    if not OUT_CSV.exists():
        print("[retry] normal_augmented.csv 없음 — 전체 루프 재시도")
        return list(range(N_LOOPS))

    df = pd.read_csv(OUT_CSV)
    loop_counts = df.groupby("loop_idx").size()
    missing = [
        loop_idx for loop_idx in range(N_LOOPS)
        if loop_counts.get(loop_idx, 0) < SCENARIOS_PER_CALL
    ]
    print(f"[retry] 부족 루프 {len(missing)}개: {missing}")
    return missing


def build_retry_jsonl(
    system_prompt: str,
    normal_df: pd.DataFrame,
    minsang_df: pd.DataFrame,
    missing_loops: list[int],
    seed: int = SEED,
) -> None:
    """원본과 동일한 rng 시퀀스를 재생하여 실패 루프만 JSONL로 저장."""
    import random
    rng = random.Random(seed)
    missing_set = set(missing_loops)
    records = []

    for loop_idx in range(N_LOOPS):
        a_seed = rng.randint(0, 9999)
        b_seed = rng.randint(0, 9999)

        if loop_idx not in missing_set:
            continue

        track_loop_idx = loop_idx // 2
        scen_offset    = (track_loop_idx * SCENARIOS_PER_CALL) % 10

        if loop_idx % 2 == 0:
            shots    = [clean_script(str(r["script"])) for r in minsang_df.sample(5, random_state=a_seed).to_dict("records")]
            track    = "A"
            all_scen = A_SCENARIOS
            all_kw   = A_KEYWORDS
        else:
            shots    = [clean_script(str(r["script"])) for r in normal_df.sample(5, random_state=b_seed).to_dict("records")]
            track    = "B"
            all_scen = B_SCENARIOS
            all_kw   = B_KEYWORDS

        indices   = [scen_offset % 10, (scen_offset + 1) % 10]
        scenarios = [all_scen[i] for i in indices]
        keywords  = [all_kw[i]   for i in indices]

        records.append({
            "custom_id": f"loop_{loop_idx:04d}_{track}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": build_single_track_prompt(track, shots, scenarios, keywords, n_per_scenario=1)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 1.0,
                "max_completion_tokens": 8192,
            },
        })

    with open(RETRY_BATCH_INPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[retry] JSONL 생성: {RETRY_BATCH_INPUT_JSONL} ({len(records)}개 요청)")


def submit_retry_batch(client: OpenAI) -> str:
    print("[retry] 파일 업로드 중...")
    with open(RETRY_BATCH_INPUT_JSONL, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    file_id = uploaded.id
    print(f"[retry] 업로드 완료: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "normal_augmented_retry"},
    )
    batch_id = batch.id
    print(f"[retry] 배치 제출 완료: {batch_id} | 상태: {batch.status}")

    meta = {"batch_id": batch_id, "file_id": file_id, "status": batch.status}
    RETRY_BATCH_META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[retry] 메타 저장: {RETRY_BATCH_META_JSON}")
    return batch_id


def collect_retry_results(client: OpenAI, batch_id: str | None = None) -> None:
    meta = json.loads(RETRY_BATCH_META_JSON.read_text(encoding="utf-8"))
    if batch_id is None:
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

    print(f"[retry] 결과 다운로드: {batch.output_file_id}")
    content = client.files.content(batch.output_file_id).content
    lines   = content.decode("utf-8").strip().split("\n")

    # 기존 CSV 로드
    existing_df = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()

    new_rows: list[dict] = []
    retried_loops: set[int] = set()
    parse_errors: list[str] = []

    for line in lines:
        result    = json.loads(line)
        custom_id = result["custom_id"]
        parts     = custom_id.split("_")
        loop_idx  = int(parts[1])
        track     = parts[2]
        retried_loops.add(loop_idx)

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
        if len(scripts) != SCENARIOS_PER_CALL:
            print(f"  [경고] {custom_id}: scripts={len(scripts)}개 ({SCENARIOS_PER_CALL}개 기대)")

        for scen_idx, script in enumerate(scripts[:SCENARIOS_PER_CALL]):
            new_rows.append({
                "script":       str(script).strip(),
                "label":        0,
                "track":        track,
                "scenario_idx": scen_idx + 1,
                "loop_idx":     loop_idx,
            })

    # 기존 CSV에서 재시도 성공한 loop_idx 제거 후 병합
    succeeded = retried_loops - {int(c.split("_")[1]) for c in parse_errors}
    if not existing_df.empty:
        existing_df = existing_df[~existing_df["loop_idx"].isin(succeeded)]

    # uid 재발급 (기존 최대값 이후부터)
    max_uid = 0
    if not existing_df.empty and "id" in existing_df.columns:
        nums = existing_df["id"].str.extract(r"(\d+)$")[0].dropna().astype(int)
        max_uid = nums.max() if not nums.empty else 0

    for i, row in enumerate(new_rows, start=1):
        row["id"] = f"normal_aug_{max_uid + i:04d}"

    retry_df = pd.DataFrame(new_rows, columns=["id", "script", "label", "track", "scenario_idx", "loop_idx"])
    merged_df = pd.concat([existing_df, retry_df], ignore_index=True)
    merged_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[retry] 병합 완료: 총 {len(merged_df)}개 → {OUT_CSV}")
    print(f"[retry] 재시도 성공: {len(succeeded)}루프 ({len(new_rows)}개 스크립트 추가)")
    if parse_errors:
        print(f"[retry] 재시도 실패: {len(parse_errors)}개: {parse_errors}")
    print(f"[retry] 트랙별:\n{merged_df['track'].value_counts().to_string()}")


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
        if len(scripts) != SCENARIOS_PER_CALL:
            print(f"  [경고] {custom_id}: scripts={len(scripts)}개 ({SCENARIOS_PER_CALL}개 기대)")

        for scen_idx, script in enumerate(scripts[:SCENARIOS_PER_CALL]):
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
        choices=["submit", "check", "collect", "cancel", "submit-retry", "check-retry", "collect-retry"],
        required=True,
        help="submit: 배치 생성·제출 | check: 상태 확인 | collect: 결과 수집 | cancel: 배치 취소 | submit-retry: 실패 루프 재제출 | check-retry: 재시도 배치 상태 확인 | collect-retry: 재시도 결과 수집·병합",
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

    elif args.mode == "submit-retry":
        print("[load] CSV 로딩...")
        normal_df  = pd.read_csv(NORMAL_CSV)
        minsang_df = pd.read_csv(MINSANG_CSV)
        system_prompt = SYSTEM_PROMPT_TXT.read_text(encoding="utf-8")
        missing = find_missing_loops()
        if not missing:
            print("[retry] 재시도할 루프 없음 — 모든 루프 완료 상태")
        else:
            build_retry_jsonl(system_prompt, normal_df, minsang_df, missing)
            submit_retry_batch(client)

    elif args.mode == "check-retry":
        meta = json.loads(RETRY_BATCH_META_JSON.read_text(encoding="utf-8"))
        check_batch(client, args.batch_id or meta["batch_id"])

    elif args.mode == "collect-retry":
        collect_retry_results(client, args.batch_id)


if __name__ == "__main__":
    main()
