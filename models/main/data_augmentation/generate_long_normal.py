"""
generate_long_normal.py

일상 대화 중 Very Long(16+ 세그먼트) 샘플 95개 생성.
OpenAI Batch API 방식 (submit → check → collect).

── 기존 대비 변경점 ─────────────────────────────────────────────────────────
  [시스템 프롬프트] "길이 800~3000자" → "길이 2500~4000자"
  [유저 프롬프트]  발화당 평균 150자 이상, 상황 설명 발화 포함, 최소 2500자
  [생성 방식]      동기 API → Batch API (submit/check/collect)

실행 방법:
  export OPENAI_API_KEY=...

  # 1단계: 배치 제출
  python generate_long_normal.py --mode submit [--n-target 95] [--seed 42]

  # 2단계: 완료 확인 (수십 분~수 시간 소요)
  python generate_long_normal.py --mode check

  # 3단계: 결과 수집 (ASR 노이즈 + segment_risks 계산)
  python generate_long_normal.py --mode collect

  # 실패/부족 시: 기존 CSV 확인 후 부족분만 재제출
  python generate_long_normal.py --mode retry

  # 배치 취소
  python generate_long_normal.py --mode cancel
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("[ERROR] pip install openai")

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent
DATA_DIR          = BASE_DIR / "output" / "4class"
OUT_DIR           = BASE_DIR / "output"
ERROR_JSON        = BASE_DIR / "error_analysis" / "error_summary.json"
TRAIN_CSV         = DATA_DIR / "train.csv"
BATCH_INPUT_JSONL = OUT_DIR / "long_normal_batch_input.jsonl"
BATCH_META_JSON   = OUT_DIR / "long_normal_batch_meta.json"
OUT_CSV           = OUT_DIR / "long_normal_augmented.csv"

MODEL             = "gpt-5.4-mini"
TOKENIZER_NAME    = "klue/roberta-base"
WINDOW_SIZE       = 64
STRIDE            = 32
N_PER_SCENARIO    = 2    # 시나리오당 생성 수 → 5×2=10개/호출
MIN_LEN           = 1000  # collect 시 최소 문자 수 필터
MIN_UTTERANCE_LEN = 150   # 긴 발화 필터: 문장 절 단위 최대 길이 기준 (설명형 확인용)

# ── 시나리오 정의 (기존 generate_normal_augmented.py와 동일) ────────────────
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

# 보이스피싱에서 자주 사칭되는 은행·카드사·기관 목록
# 시스템 프롬프트에 삽입되어 LLM이 스크립트마다 무작위로 선택하도록 유도
PHISHING_ORG_LIST = [
    "KB국민은행", "신한은행", "우리은행", "하나은행", "NH농협은행",
    "IBK기업은행", "카카오뱅크", "케이뱅크", "토스뱅크", "SC제일은행",
    "KB국민카드", "신한카드", "삼성카드", "현대카드", "롯데카드",
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

# ── 시스템 프롬프트 ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """생성하는 모든 대화는 범죄와 무관한 정상적인 상황이어야 한다.
은행·카드사·기관 이름이 등장해야 하는 경우, 아래 목록에서 스크립트마다 서로 다른 기관을 무작위로 선택하여 사용하라.
같은 배치 내 스크립트끼리 동일한 기관명이 반복되지 않도록 다양하게 분산하라.
[사용 가능한 기관 목록]
KB국민은행, 신한은행, 우리은행, 하나은행, NH농협은행, IBK기업은행, 카카오뱅크, 케이뱅크, 토스뱅크, SC제일은행, KB국민카드, 신한카드, 삼성카드, 현대카드, 롯데카드


범죄 관련 키워드가 등장하더라도 반드시 뉴스 시청, 드라마 감상, 은행 예방 안내, 보안 교육 등 일상적·합법적 맥락에서만 사용해야 한다.

[트랙 구분]
- 트랙 A (콜센터): 은행, 카드사, 통신사 등 실제 콜센터 상담 통화 스타일
  - 상담사와 고객 사이의 전화 상담 대화
  - 전문적이고 정중하지만 친근한 말투
  - 확인 절차, 안내, 해결 등의 흐름
- 트랙 B (일상대화): 가족, 친구, 동료 사이의 일상 전화 통화 스타일
  - 격식 없고 자연스러운 구어체
  - 감탄사, 추임새, 반응 표현이 풍부
  - 이야기를 주고받는 대화 흐름

[작성 규칙]
- 발신자/수신자 라벨 없이 연속 텍스트로 작성
- 한국어 구어체, 실제 전화 통화 녹취록처럼 작성
- 줄바꿈 문자 사용 금지 (한 줄 연속 텍스트로 작성)
- 길이 2500~4000자 (항목마다 길이를 다양하게)
- 영어 혼용 금지, 외래어는 한국어 발음 표기로만 사용
- 전화 시작 패턴이 일정하지 않고 다양해야 함
- 마침표(.), 쉼표(,), 물음표(?), 말줄임표(...) 등의 문장 부호를 적절히 사용하여 대화의 호흡, 억양, 화자 교대를 자연스럽게 표현해야 함
- 감정/은어 억제: 인터넷 용어(키키, 하하, ㅠㅠ, ㅎㅎ, ㅋㅋ 등) 및 이모티콘 사용 금지
- 기계적 서두 억제: "안녕하세요", "상담원 OOO입니다", "무엇을 도와드릴까요" 같은 정형화된 콜센터 도입부 사용 금지
- 리액션 억제: "아, 그렇군요", "진짜요?", "헐" 같은 지나치게 짧고 작위적인 리액션 사용 금지

[구어체 필수 반영]
- "음", "어", "저", "그러니까", "아" 같은 망설임 표현을 자연스럽게 삽입
- 말이 끊기거나 다시 시작하는 자기 수정 패턴 포함
- 필수 키워드는 자연스러운 문맥 속에 반드시 포함되어야 함

[핵심 제약 사항]
- 각 발화는 2~4문장 이상의 충분한 길이로 작성하라. 짧은 리액션("네", "아", "그렇군요")만 이어지는 핑퐁식 교환은 금지한다.
- 상담원(또는 발신자)은 상품·절차·상황을 충분히 설명하고, 고객(또는 수신자)은 자신의 상황을 구체적으로 길게 이야기하는 발화를 포함하라.
- 전체 대화는 최소 2500자를 유지하라. 발화 교환은 15회 이상, 발화당 평균 150자 이상을 목표로 한다.
- 대화 중 한 화자가 300자 이상의 긴 설명을 이어가는 발화를 최소 2회 이상 포함하라. 예를 들어 상담원이 상품 조건을 상세히 풀어 설명하거나, 고객이 자신의 상황을 길게 서술하는 발화가 이에 해당한다. 이는 실제 전화 통화에서 한쪽이 그럴듯한 설명을 길게 늘어놓는 자연스러운 패턴을 반영한 것이다.
- 아래 어휘들이 보이스피싱 맥락이 아닌 정상적·합법적 맥락에서 자연스럽게 등장하도록 의도적으로 포함하라. 이는 모델이 특정 단어만으로 판단하지 못하도록 하기 위함이다.
  · 금융 어휘: 계좌번호, 이체, 송금, 한도, 금리, 대출, 명의, 신용점수, 대환
  · 수사·기관 어휘: 수사, 압수, 동결, 검사, 형사, 명의도용, 대포통장
  · 긴박감 표현: 오늘 안에, 즉시, 긴급, 마감, 바로 처리
  단, 위 어휘들은 반드시 정상적인 문맥(뉴스 이야기, 실제 은행 업무, 피해 신고, 지인 경고 등) 안에서만 사용해야 한다.
- 키워드 고밀도 유지: 필수 키워드를 대화 전반에 걸쳐 각각 2~3회 이상 반복 사용하라.
- 한 주제를 끝까지 유지하되, 자연스럽게 세부 사항을 심화하며 대화를 이어가라.

[출력 형식]
반드시 아래 JSON 형식으로만 응답하라. 다른 설명 텍스트는 일절 포함하지 말 것.
{"scripts": ["스크립트1", "스크립트2", ...]}"""


# ── 유저 프롬프트 구성 ────────────────────────────────────────────────────────
def build_user_prompt(track: str, shots: list, scenarios: list,
                      keywords: list, n_per_scenario: int) -> str:
    track_label = "A (콜센터)" if track == "A" else "B (일상대화)"
    n_total = len(scenarios) * n_per_scenario
    lines = []

    lines.append(f"[트랙 {track_label} 예시 대화]")
    for i, s in enumerate(shots, 1):
        lines.append(f"예시 {i}: {s[:600]}")

    lines.append("")
    lines.append("[생성 요청]")
    lines.append(
        f"아래 시나리오에 맞게 트랙 {track_label} 스크립트를 각 {n_per_scenario}개씩, "
        f"총 {n_total}개 생성하라."
    )
    lines.append("반드시 필수 키워드를 자연스럽게 포함하되, 범죄 맥락이 아닌 정상적인 맥락이어야 한다.")
    lines.append(
        "각 스크립트는 발화당 평균 150자 이상의 충분한 길이로 작성하라. "
        "상담원(또는 발신자)은 상품·절차·상황을 2~4문장으로 상세히 설명하고, "
        "고객(또는 수신자)은 자신의 상황을 구체적으로 이야기하는 발화를 반드시 포함하라. "
        "전체 대화는 최소 2500자, 발화 교환 15회 이상을 반드시 충족해야 한다."
    )

    lines.append("")
    lines.append("[시나리오 목록]")
    for i, (scen, kw) in enumerate(zip(scenarios, keywords), 1):
        lines.append(f"{i}. 시나리오: {scen}")
        lines.append(f"   필수 키워드: {kw}")

    lines.append("")
    lines.append("【출력 규칙】")
    lines.append(f'- "scripts" 배열에 시나리오 순서대로 각 {n_per_scenario}개씩, 총 정확히 {n_total}개 출력하라.')
    lines.append('- 배열 순서: 시나리오1의 1번째, 시나리오1의 2번째, 시나리오2의 1번째, 시나리오2의 2번째, ...')
    example_items = ", ".join([f'"스크립트{i}"' for i in range(1, n_total + 1)])
    lines.append(f'- 반드시 JSON만 출력: {{"scripts": [{example_items}]}}')

    return "\n".join(lines)


# ── few-shot 로드: nseg >= 16인 정상 샘플 우선 ──────────────────────────────
def load_fewshot_pool(train_csv: Path) -> dict:
    rows = []
    with train_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat  = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if cat not in {"상담 대화", "일상 대화"} or not text:
                continue
            sr = row.get("segment_risks", "")
            try:
                nseg = len(json.loads(sr)) if sr else 0
            except Exception:
                nseg = 0
            src   = (row.get("source") or "")
            track = "A" if "minsang" in src or "상담" in cat else "B"
            rows.append({"text": text, "nseg": nseg, "track": track})

    pool = {"A": [], "B": []}
    for track in ("A", "B"):
        track_rows = [r for r in rows if r["track"] == track]
        long_rows  = [r["text"] for r in track_rows if r["nseg"] >= 16]
        all_rows   = [r["text"] for r in track_rows]
        pool[track] = long_rows if len(long_rows) >= 5 else all_rows
        print(f"  Track {track}: {len(pool[track])}개 후보 (nseg>=16: {len(long_rows)}개)")
    return pool


# ── 배치 JSONL 생성 ──────────────────────────────────────────────────────────
def build_batch_jsonl(n_target: int, seed: int) -> int:
    print("[few-shot] 긴 정상 대화 샘플 로드 중...")
    pool = load_fewshot_pool(TRAIN_CSV)

    rng  = random.Random(seed)
    n_A  = n_target // 2 + n_target % 2   # 48
    n_B  = n_target // 2                   # 47

    per_call   = N_PER_SCENARIO  # 1×2=2 (시나리오 1개씩 호출)
    n_calls_A  = (n_A + per_call - 1) // per_call + 5   # 버퍼 +5
    n_calls_B  = (n_B + per_call - 1) // per_call + 5

    records = []
    for call_idx in range(n_calls_A):
        scen_idx = call_idx % len(A_SCENARIOS)
        shots = rng.sample(pool["A"], min(5, len(pool["A"])))
        records.append({
            "custom_id": f"long_normal_A_{call_idx:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(
                        "A", shots,
                        [A_SCENARIOS[scen_idx]], [A_KEYWORDS[scen_idx]],
                        N_PER_SCENARIO)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 1.0,
                "max_completion_tokens": 8192,
            },
        })

    for call_idx in range(n_calls_B):
        scen_idx = call_idx % len(B_SCENARIOS)
        shots = rng.sample(pool["B"], min(5, len(pool["B"])))
        records.append({
            "custom_id": f"long_normal_B_{call_idx:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(
                        "B", shots,
                        [B_SCENARIOS[scen_idx]], [B_KEYWORDS[scen_idx]],
                        N_PER_SCENARIO)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 1.0,
                "max_completion_tokens": 8192,
            },
        })

    with BATCH_INPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[batch] JSONL 생성: {len(records)}개 요청 (A={n_calls_A}회, B={n_calls_B}회)")
    print(f"[batch] 예상 생성: A 최대 {n_calls_A * per_call}개, B 최대 {n_calls_B * per_call}개")
    print(f"[batch] 목표:     A={n_A}개, B={n_B}개")
    return len(records)


# ── 배치 제출 ────────────────────────────────────────────────────────────────
def submit_batch(client: OpenAI, n_target: int, seed: int) -> None:
    build_batch_jsonl(n_target, seed)

    print("\n[batch] 파일 업로드 중...")
    with BATCH_INPUT_JSONL.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"[batch] 업로드 완료: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"long_normal_aug_{n_target}"},
    )
    print(f"[batch] 제출 완료: {batch.id}  상태: {batch.status}")

    meta = {
        "batch_id": batch.id,
        "file_id":  uploaded.id,
        "n_target": n_target,
        "seed":     seed,
        "status":   batch.status,
    }
    BATCH_META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[batch] 메타 저장: {BATCH_META_JSON}")


# ── 배치 상태 확인 ────────────────────────────────────────────────────────────
def check_batch(client: OpenAI, batch_id: str = None) -> None:
    if batch_id is None:
        meta     = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]
    batch  = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"[batch] ID   : {batch_id}")
    print(f"[batch] 상태 : {batch.status}")
    print(f"[batch] 완료 : {counts.completed} / {counts.total}  실패: {counts.failed}")
    if batch.output_file_id:
        print(f"[batch] 출력 : {batch.output_file_id}")


# ── 배치 취소 ────────────────────────────────────────────────────────────────
def cancel_batch(client: OpenAI, batch_id: str = None) -> None:
    if batch_id is None:
        meta     = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]
    batch = client.batches.cancel(batch_id)
    print(f"[batch] 취소 요청: {batch_id}  상태: {batch.status}")


# ── 재제출: 기존 CSV 확인 후 부족분만 ────────────────────────────────────────
def retry_batch(client: OpenAI, n_target: int, seed: int, noise_scale: float = 1.0) -> None:
    """기존 long_normal_augmented.csv를 확인해 부족한 만큼만 추가 제출."""
    collected = 0
    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8-sig") as f:
            collected = sum(1 for _ in csv.DictReader(f))
        print(f"[retry] 기존 수집: {collected}개 / 목표 {n_target}개")
    else:
        print(f"[retry] 기존 CSV 없음. 전체 {n_target}개 새로 제출.")

    remaining = n_target - collected
    if remaining <= 0:
        print(f"[retry] 이미 목표 달성 ({collected}개). 재제출 불필요.")
        return

    print(f"[retry] {remaining}개 추가 필요 → 새 배치 제출 (seed={seed + 1})")
    submit_batch(client, remaining, seed + 1)   # seed+1로 다양성 확보
    # collect는 별도로 실행: python generate_long_normal.py --mode collect --noise-scale <값>


# ── 긴 발화 필터 ─────────────────────────────────────────────────────────────
def _max_segment_len(text: str) -> int:
    """마침표·물음표·느낌표로 구분한 절 중 최대 길이 반환 (긴 발화 여부 추정).
    말줄임표(...) 내부 마침표는 분리하지 않도록 처리."""
    parts = re.split(r'(?<![.])[.?!](?![.])', text)
    return max((len(p.strip()) for p in parts if p.strip()), default=0)


# ── ASR 노이즈 ───────────────────────────────────────────────────────────────
INITIALS = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
MEDIALS  = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
FINALS   = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
SIMILAR_INITIAL = {"ㄱ":["ㅋ"],"ㅋ":["ㄱ"],"ㄷ":["ㅌ"],"ㅌ":["ㄷ"],"ㅂ":["ㅍ"],"ㅍ":["ㅂ"],"ㅈ":["ㅊ"],"ㅊ":["ㅈ"],"ㅅ":["ㅆ"],"ㅆ":["ㅅ"]}
SIMILAR_MEDIAL  = {"ㅐ":["ㅔ"],"ㅔ":["ㅐ"],"ㅏ":["ㅓ"],"ㅓ":["ㅏ"],"ㅗ":["ㅜ"],"ㅜ":["ㅗ"],"ㅡ":["ㅜ","ㅗ"],"ㅣ":["ㅟ","ㅢ"]}
SIMILAR_FINAL   = {"ㄱ":["ㅋ"],"ㅋ":["ㄱ"],"ㄷ":["ㅌ"],"ㅌ":["ㄷ"],"ㅂ":["ㅍ"],"ㅍ":["ㅂ"],"ㅅ":["ㅆ"],"ㅆ":["ㅅ"]}
INSERT_TOKENS   = ["음","어","아","그","저","네","요"]

def _decompose(ch):
    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        return None
    s = code - 0xAC00
    return s // 588, (s % 588) // 28, s % 28

def _compose(i, m, f):
    return chr(0xAC00 + i * 588 + m * 28 + f)

def _substitute(ch, rng):
    dec = _decompose(ch)
    if dec is None:
        return rng.choice(INSERT_TOKENS)
    i, m, f = dec
    cands = []
    if INITIALS[i] in SIMILAR_INITIAL:
        for ni in SIMILAR_INITIAL[INITIALS[i]]:
            cands.append((INITIALS.index(ni), m, f))
    if MEDIALS[m] in SIMILAR_MEDIAL:
        for nm in SIMILAR_MEDIAL[MEDIALS[m]]:
            cands.append((i, MEDIALS.index(nm), f))
    if FINALS[f] in SIMILAR_FINAL:
        for nf in SIMILAR_FINAL[FINALS[f]]:
            cands.append((i, m, FINALS.index(nf)))
    if not cands:
        return ch
    ni, nm, nf = rng.choice(cands)
    return _compose(ni, nm, nf)

def apply_asr_noise(text: str, probs: dict, rng: random.Random) -> str:
    p_sub   = min(max(probs.get("substitution", 0.0), 0.0), 0.5)
    p_del   = min(max(probs.get("deletion",     0.0), 0.0), 0.5)
    p_ins   = min(max(probs.get("insertion",    0.0), 0.0), 0.5)
    p_space = min(max(probs.get("whitespace",   0.0), 0.0), 0.5)
    out = []
    for ch in text:
        if ch == " ":
            if rng.random() < p_space:
                continue
            out.append(ch)
            continue
        if rng.random() < p_del:
            continue
        out.append(_substitute(ch, rng) if rng.random() < p_sub else ch)
        if rng.random() < p_ins:
            out.append(rng.choice(INSERT_TOKENS))
        if rng.random() < p_space / 2:
            out.append(" ")
    return "".join(out)


# ── segment_risks 계산 ────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")

def compute_segment_risks(text: str, tokenizer, window_size: int, stride: int) -> list:
    cleaned = _PUNCT_RE.sub("", text)
    cleaned = re.sub(r" +", " ", cleaned).strip()
    if not cleaned:
        return []
    enc = tokenizer(cleaned, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
    n = len(enc["offset_mapping"])
    if n == 0:
        return []
    tok_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    risks, pos = [], 0
    while pos < n:
        end = min(pos + window_size, n) - 1
        while end + 1 < n and tok_strs[end + 1].startswith("##"):
            end += 1
        risks.append(1)
        if end + 1 >= n:
            break
        next_pos = pos + stride
        while next_pos < n and tok_strs[next_pos].startswith("##"):
            next_pos += 1
        if next_pos >= n:
            break
        pos = next_pos
    return risks


# ── 결과 수집 ────────────────────────────────────────────────────────────────
def collect_results(client: OpenAI, batch_id: str = None,
                    n_target: int = 95, seed: int = 42,
                    noise_scale: float = 1.0) -> None:
    if batch_id is None:
        meta     = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]
        n_target = meta.get("n_target", n_target)
        seed     = meta.get("seed", seed)

    print("[collect] 배치 완료 대기 중...")
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  상태: {batch.status} | {batch.request_counts.completed}/{batch.request_counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        if batch.output_file_id:
            print(f"[collect] 배치 상태: {batch.status} — 부분 결과만 수집합니다.")
        else:
            print(f"[collect] 배치 실패: {batch.status}. 수집 가능한 출력 없음.")
            print("[collect] → python generate_long_normal.py --mode retry 로 재제출하세요.")
            return

    print(f"[collect] 결과 다운로드: {batch.output_file_id}")
    content = client.files.content(batch.output_file_id).content
    lines   = content.decode("utf-8").strip().split("\n")

    # 트랙별 스크립트 수집
    scripts_by_track = {"A": [], "B": []}
    for line in lines:
        result    = json.loads(line)
        custom_id = result["custom_id"]   # long_normal_A_0000
        track     = custom_id.split("_")[2]   # A 또는 B

        if result.get("error"):
            print(f"  [오류] {custom_id}: {result['error']}")
            continue

        raw = result["response"]["body"]["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}

        for s in parsed.get("scripts", []):
            s = str(s).strip()
            if len(s) < MIN_LEN:
                continue
            max_seg = _max_segment_len(s)
            if max_seg < MIN_UTTERANCE_LEN:
                print(f"  [필터] {custom_id}: 긴 발화 없음 (최대 절={max_seg}자) — 제외")
                continue
            scripts_by_track[track].append(s)

    n_A = n_target // 2 + n_target % 2
    n_B = n_target // 2
    print(f"\n[collect] 수집: A={len(scripts_by_track['A'])}개, B={len(scripts_by_track['B'])}개")
    print(f"[collect] 목표: A={n_A}개, B={n_B}개")

    # 토크나이저 + ASR 노이즈 로드
    try:
        from transformers import AutoTokenizer
        print(f"\n[tokenizer] {TOKENIZER_NAME} 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        print("  완료")
    except ImportError:
        raise SystemExit("[ERROR] pip install transformers")

    with ERROR_JSON.open("r", encoding="utf-8") as f:
        noise_probs = json.load(f)
    if noise_scale != 1.0:
        for key in ("substitution", "deletion", "insertion", "whitespace"):
            noise_probs[key] = noise_probs.get(key, 0.0) * noise_scale
        print(f"[collect] 노이즈 스케일: {noise_scale}  "
              f"(sub={noise_probs['substitution']:.3f}  "
              f"del={noise_probs['deletion']:.3f}  "
              f"ins={noise_probs['insertion']:.3f}  "
              f"ws={noise_probs['whitespace']:.3f})")
    rng = random.Random(seed)

    # 기존 CSV 이어쓰기 지원
    existing_rows = []
    next_uid = 1
    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8-sig") as f:
            existing_rows = list(csv.DictReader(f))
        if existing_rows:
            last_id  = existing_rows[-1].get("id", "long_normal_aug_0000")
            next_uid = int(last_id.split("_")[-1]) + 1
        print(f"[collect] 기존 CSV: {len(existing_rows)}개 (uid {next_uid}부터 이어쓰기)")

    all_rows, uid = [], next_uid
    for track, n_take in [("A", n_A), ("B", n_B)]:
        taken = scripts_by_track[track][:n_take]
        if len(taken) < n_take:
            print(f"  [경고] Track {track}: {len(taken)}개만 확보 (목표 {n_take}개)")
        for raw_text in taken:
            noised = apply_asr_noise(raw_text, noise_probs, rng)
            risks  = compute_segment_risks(noised, tokenizer, WINDOW_SIZE, STRIDE)
            nseg   = len(risks)
            row = {
                "id":            f"long_normal_aug_{uid:04d}",
                "text":          noised,
                "label":         "0" if track == "A" else "1",
                "category":      "상담 대화" if track == "A" else "일상 대화",
                "source":        "llm_long_aug",
                "filename":      "",
                "segment_risks": json.dumps(risks, ensure_ascii=False),
            }
            all_rows.append(row)
            uid += 1
            print(f"  [OK] {row['id']}  track={track}  nseg={nseg}  len={len(noised)}자")

    final_rows = existing_rows + all_rows
    FIELDNAMES = ["id", "text", "label", "category", "source", "filename", "segment_risks"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    total = len(final_rows)
    vl    = sum(1 for r in final_rows if len(json.loads(r["segment_risks"])) >= 16)
    print(f"\n[완료] {OUT_CSV}")
    print(f"  기존 {len(existing_rows)}개 + 신규 {len(all_rows)}개 = 총 {total}개")
    print(f"  16+ 세그먼트: {vl}개 ({vl / total * 100:.1f}%)")
    if total < n_target:
        print(f"\n  [경고] 목표 {n_target}개 미달 ({total}개 확보).")
        print(f"  → python generate_long_normal.py --mode retry 로 부족분({n_target - total}개) 재제출하세요.")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["submit", "check", "collect", "cancel", "retry"],
                        required=True,
                        help="submit: 배치 제출 | check: 상태 확인 | collect: 결과 수집 | "
                             "cancel: 배치 취소 | retry: 부족분 재제출")
    parser.add_argument("--n-target", type=int, default=95, help="생성할 총 샘플 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-id", default=None, help="check/collect 시 batch ID 직접 지정")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="ASR 노이즈 스케일 (0.5=절반, 1.0=원본, 기본 1.0). collect/retry 모드에 적용")
    args = parser.parse_args()

    client = OpenAI()

    if args.mode == "submit":
        submit_batch(client, args.n_target, args.seed)
    elif args.mode == "check":
        check_batch(client, args.batch_id)
    elif args.mode == "collect":
        collect_results(client, args.batch_id, args.n_target, args.seed, args.noise_scale)
    elif args.mode == "cancel":
        cancel_batch(client, args.batch_id)
    elif args.mode == "retry":
        retry_batch(client, args.n_target, args.seed, args.noise_scale)


if __name__ == "__main__":
    main()
