"""
generate_short_phishing.py

피싱 카테고리 아웃라이어 제거분 보충: 짧은(5~12 세그먼트) 피싱 대화 생성.
  - 대출 사기형:    11개
  - 수사기관 사칭형: 19개
  합계: 30개

OpenAI Batch API 방식 (submit → check → collect).

── 기존 대비 변경점 ─────────────────────────────────────────────────────────
  [시스템 프롬프트] "길이 800~2000자" → "길이 400~900자" + 발화 교환 5~8회 이내
  [출력 포맷]      [변형N] 태그 → JSON {"scripts": [...]}
  [생성 방식]      동기 API → Batch API (submit/check/collect)
  [few-shot 선택]  nseg <= 12인 짧은 샘플 우선

실행 방법:
  export OPENAI_API_KEY=...

  # 1단계: 배치 제출
  python generate_short_phishing.py --mode submit [--seed 42]

  # 2단계: 완료 확인
  python generate_short_phishing.py --mode check

  # 3단계: 결과 수집 (ASR 노이즈 + segment_risks 계산)
  python generate_short_phishing.py --mode collect

  # 실패/부족 시: 기존 CSV 확인 후 카테고리별 부족분만 재제출
  python generate_short_phishing.py --mode retry

  # 배치 취소
  python generate_short_phishing.py --mode cancel
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
BATCH_INPUT_JSONL = OUT_DIR / "short_phishing_batch_input.jsonl"
BATCH_META_JSON   = OUT_DIR / "short_phishing_batch_meta.json"
OUT_CSV           = OUT_DIR / "short_phishing_augmented.csv"

MODEL          = "gpt-5.4-mini"
TOKENIZER_NAME = "klue/roberta-base"
WINDOW_SIZE    = 64
STRIDE         = 32
PER_CALL       = 2    # 호출당 생성 수

# 생성 목표 및 위험도
TARGETS  = {"대출 사기형": 11, "수사기관 사칭형": 19}
RISK_MAP = {"대출 사기형": 2,  "수사기관 사칭형": 3}
CAT_KEY  = {"대출 사기형": "loan", "수사기관 사칭형": "inv"}
KEY_CAT  = {"loan": "대출 사기형", "inv": "수사기관 사칭형"}

# ── 카테고리 프롬프트 (category_prompts.json과 동일) ────────────────────────
CATEGORY_PROMPTS = {
    "대출 사기형": """\
[카테고리: 대출 사기형]

[시나리오 핵심]
- 은행/저축은행 상담사 사칭
- 저금리 대환대출 또는 한도 안내
- 부결/신용점수 문제로 압박
- 선납금/보증금/수수료 입금 유도
- 마감 임박 강조

[역할 고정]
- 통화 시작은 반드시 사칭 금융기관 상담사가 주도
- 피해자 문의 전화처럼 시작 금지

[기관명 제한]
- 농협, 국민은행, 신한은행, 우리은행, 하나은행, 기업은행, 카카오뱅크, 토스뱅크, 케이뱅크, KB저축은행, SBI저축은행, 웰컴저축은행, OK저축은행, 햇살론, 삼성카드

[필수 요소]
- 금리/한도 등 구체 수치 안내
- 부결 또는 신용점수 문제 언급
- 편법 대환대출 제안
- 선납금 요구
- 마감 긴박감 조성

[구어체 특성]
- 상담원 어투는 친근하고 빠른 말투, 피해자 반응에 맞춰 자연스럽게 이어감
- 수치 설명 중 수정하거나 재확인하는 패턴 포함
- 피해자 망설임에 빠르게 대응하며 끊는 패턴 포함""",

    "수사기관 사칭형": """\
[카테고리: 수사기관 사칭형]

[시나리오 핵심]
- 검찰/수사관/금감원 사칭
- 피해자 명의 대포통장 또는 범죄 연루 통보
- 계좌 동결/압수/체포 위협
- 제3자 연락 차단과 비밀 유지 강요
- 최종적으로 계좌 정보 확인 또는 이체 유도
- 계좌번호, 비밀번호, 주민등록번호 등 개인정보 요구

[역할 고정]
- 통화 시작은 반드시 사칭 수사기관 측이 주도
- 피해자 문의 전화처럼 시작 금지

[기관명 제한]
- 사칭 기관: 검찰청, 서울중앙지방검찰청, 지방검찰청, 금융감독원, 경찰청
- 계좌 기관: 농협, 하나은행, 국민은행, 기업은행

[필수 요소]
- 피해자 명의 대포통장 발견 또는 명의도용 통보
- 수사관/검사 신분 강조
- 계좌 동결/압수 경고
- 긴박감 조성 표현
- 제3자 차단 요구
- 계좌번호·비밀번호·주민등록번호 등 개인정보 요구

[구어체 특성]
- 수사관 어투는 딱딱하지만 설명 중 간헐적 망설임 포함
- 피해자를 압박할 때 같은 말을 반복 강조
- 피해자 반박에 끊고 다시 설명하는 패턴 포함""",
}

# ── 시스템 프롬프트 ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """너는 보이스피싱 탐지 모델 학습을 위한 한국어 보이스 피싱 대화 STT 텍스트 생성기다.
출력은 실제 STT 전사처럼 화자 구분 없이 이어진 단일 텍스트여야 한다.

[작성 규칙]
- 발신자/수신자 라벨 없이 연속 텍스트로 작성
- 한국어 구어체, 실제 전화 통화 녹취록처럼 작성
- 줄바꿈 문자 사용 금지 (한 줄 연속 텍스트)
- 길이 1000자 내외 (변형마다 길이를 다양하게)
- 발화 교환 최소 8회 이상
- 영어 혼용 금지, 외래어는 한국어 발음 표기로만 사용
- 통화 시작 주도자는 반드시 피싱 가해자(사칭 상담원/수사관)여야 함
- 피해자가 먼저 문의 전화한 상황으로 시작하면 안 됨
- 전화 시작 패턴이 일정하지 않고 다양해야 함

[구어체 필수 반영]
- "음", "어", "저", "그러니까", "잠깐만요", "아" 같은 망설임 표현을 자연스럽게 삽입
- 말이 끊기거나 다시 시작하는 자기 수정 패턴 포함
- 단어나 어절 반복 (예: "지금 지금 바로 처리하셔야")
- 피해자 응답을 유추할 수 있는 가해자 반응 포함
- 마침표(.), 쉼표(,), 물음표(?), 느낌표(!) 등 문장부호는 자연스럽게 사용

[발화 길이 제약]
- 한 화자가 연속으로 발화하는 길이는 200자 이내로 제한한다.
- 가해자가 일방적으로 길게 설명하는 독백 형태는 금지한다. 피해자의 반응·질문·반박이 중간중간 자연스럽게 끼어들어야 한다.

[few-shot 활용 원칙]
- 제공된 예시는 말투·구어체 특성·화자 교대 패턴 등 형식만 참고한다.
- 예시의 내용(시나리오·소재·전개)은 절대 따라 하지 않는다.
- 실제 생성 내용은 반드시 주어진 카테고리 프롬프트의 시나리오 핵심을 따른다.

[출력 형식]
반드시 아래 JSON 형식으로만 응답하라. 다른 설명 텍스트는 일절 포함하지 말 것.
{"scripts": ["스크립트1", "스크립트2", ...]}"""


# ── few-shot 로드: nseg <= 12인 피싱 샘플 우선 ──────────────────────────────
def load_fewshot_pool(train_csv: Path) -> dict:
    rows_by_cat = {}
    with train_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cat  = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if cat not in TARGETS or not text:
                continue
            sr = row.get("segment_risks", "")
            try:
                nseg = len(json.loads(sr)) if sr else 0
            except Exception:
                nseg = 0
            rows_by_cat.setdefault(cat, []).append({"text": text, "nseg": nseg})

    pool = {}
    for cat, rows in rows_by_cat.items():
        short_rows = [r["text"] for r in rows if r["nseg"] <= 12]
        all_rows   = [r["text"] for r in rows]
        pool[cat]  = short_rows if len(short_rows) >= 5 else all_rows
        print(f"  [{cat}] few-shot 풀: {len(pool[cat])}개 (nseg<=12: {len(short_rows)}개)")
    return pool


# ── 유저 프롬프트 구성 ────────────────────────────────────────────────────────
def build_user_prompt(category: str, few_shots: list, n_generate: int) -> str:
    cat_prompt = CATEGORY_PROMPTS[category]
    examples   = "\n\n".join(
        f"[실제 STT 예시 {i+1}]\n{s[:300]}"
        for i, s in enumerate(few_shots)
    )
    example_items = ", ".join([f'"스크립트{i}"' for i in range(1, n_generate + 1)])
    return (
        f"{cat_prompt}\n\n"
        f"[말투 참고용 STT 예시 — 내용 모방 금지]\n{examples}\n\n"
        f"위 예시는 말투·망설임·반복·말 끊김 등 구어체 음성 전사 특성만 참고하라. 예시의 내용·소재·전개는 따라 하지 않는다.\n"
        f"생성 내용은 반드시 위 카테고리 프롬프트의 [시나리오 핵심]과 [필수 요소]를 따른다.\n\n"
        f"위 조건으로 보이스피싱 STT 전사 텍스트를 {n_generate}개 생성하라.\n"
        f"각 생성물은 발화 교환 최소 8회 이상, 1000자 내외로 작성하라.\n"
        "변형마다 서로 다른 시작 표현과 다양한 시나리오 전개를 사용하라.\n"
        f"반드시 JSON만 출력: {{\"scripts\": [{example_items}]}}"
    )


def looks_victim_initiated(text: str) -> bool:
    head = text[:140].replace(" ", "")
    return any(t in head for t in [
        "제가대출문의", "대출문의드리려고", "제가신청",
        "문의드리려고전화", "상담받으려고전화", "제가먼저전화",
    ])


# ── 배치 JSONL 생성 ──────────────────────────────────────────────────────────
def build_batch_jsonl(seed: int) -> int:
    print("[few-shot] 짧은 피싱 샘플 로드 중...")
    pool = load_fewshot_pool(TRAIN_CSV)

    rng     = random.Random(seed)
    records = []

    for category, n_target in TARGETS.items():
        cat_pool = pool.get(category, [])
        cat_key  = CAT_KEY[category]
        # 버퍼: 목표의 3배 호출
        n_calls  = (n_target + PER_CALL - 1) // PER_CALL + 3

        for call_idx in range(n_calls):
            shots = rng.sample(cat_pool, min(5, len(cat_pool)))
            records.append({
                "custom_id": f"short_phishing_{cat_key}_{call_idx:04d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(category, shots, PER_CALL)},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 1.0,
                    "max_completion_tokens": 4096,
                },
            })
        print(f"  [{category}] {n_calls}회 요청 (최대 {n_calls * PER_CALL}개 생성, 목표 {n_target}개)")

    with BATCH_INPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[batch] JSONL 생성: {len(records)}개 요청")
    return len(records)


# ── 배치 취소 ────────────────────────────────────────────────────────────────
def cancel_batch(client: OpenAI, batch_id: str = None) -> None:
    if batch_id is None:
        meta     = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]
    batch = client.batches.cancel(batch_id)
    print(f"[batch] 취소 요청: {batch_id}  상태: {batch.status}")


# ── 재제출: 카테고리별 부족분만 ──────────────────────────────────────────────
def retry_batch(client: OpenAI, seed: int, noise_scale: float = 1.0) -> None:
    """기존 CSV의 카테고리별 수집 현황을 확인해 부족분만 재제출."""
    # 기존 수집 현황 파악
    collected = {cat: 0 for cat in TARGETS}
    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cat = row.get("category", "").strip()
                if cat in collected:
                    collected[cat] += 1
        print(f"[retry] 기존 수집 현황:")
        for cat, cnt in collected.items():
            print(f"  {cat}: {cnt}개 / 목표 {TARGETS[cat]}개")
    else:
        print("[retry] 기존 CSV 없음. 전체 재제출.")

    remaining = {cat: max(0, TARGETS[cat] - collected[cat]) for cat in TARGETS}
    if all(v == 0 for v in remaining.values()):
        print("[retry] 모든 카테고리 목표 달성. 재제출 불필요.")
        return

    for cat, rem in remaining.items():
        if rem > 0:
            print(f"[retry] {cat}: {rem}개 추가 필요")

    # 부족분만 재제출 (TARGETS를 임시로 재정의)
    pool = load_fewshot_pool(TRAIN_CSV)
    rng  = random.Random(seed + 1)   # seed+1로 다양성 확보
    records = []

    for category, rem in remaining.items():
        if rem == 0:
            continue
        cat_pool = pool.get(category, [])
        cat_key  = CAT_KEY[category]
        n_calls  = (rem + PER_CALL - 1) // PER_CALL + 2   # 버퍼 +2

        for call_idx in range(n_calls):
            shots = rng.sample(cat_pool, min(5, len(cat_pool)))
            records.append({
                "custom_id": f"short_phishing_{cat_key}_retry_{call_idx:04d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(category, shots, PER_CALL)},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 1.0,
                    "max_completion_tokens": 4096,
                },
            })
        print(f"  [{category}] {n_calls}회 요청 (최대 {n_calls * PER_CALL}개, 목표 {rem}개)")

    with BATCH_INPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[retry] JSONL 생성: {len(records)}개 요청")
    print("[retry] 파일 업로드 중...")
    with BATCH_INPUT_JSONL.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "short_phishing_aug_retry"},
    )
    print(f"[retry] 제출 완료: {batch.id}  상태: {batch.status}")

    meta = {
        "batch_id": batch.id,
        "file_id":  uploaded.id,
        "seed":     seed + 1,
        "status":   batch.status,
        "is_retry": True,
        "remaining": remaining,
    }
    BATCH_META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[retry] 메타 저장: {BATCH_META_JSON}")


# ── 배치 제출 ────────────────────────────────────────────────────────────────
def submit_batch(client: OpenAI, seed: int) -> None:
    build_batch_jsonl(seed)

    print("\n[batch] 파일 업로드 중...")
    with BATCH_INPUT_JSONL.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"[batch] 업로드 완료: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "short_phishing_aug_30"},
    )
    print(f"[batch] 제출 완료: {batch.id}  상태: {batch.status}")

    meta = {
        "batch_id": batch.id,
        "file_id":  uploaded.id,
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


# ── segment_risks 계산 (피싱 위험도는 카테고리 고정값) ──────────────────────
_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")

def compute_segment_risks(text: str, tokenizer, risk_val: int,
                           window_size: int, stride: int) -> list:
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
        risks.append(risk_val)
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
def collect_results(client: OpenAI, batch_id: str = None, seed: int = 42, noise_scale: float = 1.0) -> None:
    if batch_id is None:
        meta     = json.loads(BATCH_META_JSON.read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]
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
            print("[collect] → python generate_short_phishing.py --mode retry 로 재제출하세요.")
            return

    print(f"[collect] 결과 다운로드: {batch.output_file_id}")
    content = client.files.content(batch.output_file_id).content
    lines   = content.decode("utf-8").strip().split("\n")

    # 카테고리별 스크립트 수집
    scripts_by_cat = {cat: [] for cat in TARGETS}
    for line in lines:
        result    = json.loads(line)
        custom_id = result["custom_id"]   # short_phishing_loan_0000
        cat_key   = custom_id.split("_")[2]   # loan 또는 inv
        category  = KEY_CAT.get(cat_key)
        if category is None:
            print(f"  [경고] 알 수 없는 cat_key: {cat_key}")
            continue

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
            if 150 <= len(s) <= 1200 and not looks_victim_initiated(s):
                scripts_by_cat[category].append(s)

    for cat, n_target in TARGETS.items():
        print(f"  [{cat}] 수집: {len(scripts_by_cat[cat])}개 (목표 {n_target}개)")

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
            last_id  = existing_rows[-1].get("id", "short_phishing_aug_0000")
            next_uid = int(last_id.split("_")[-1]) + 1
        # 기존 카테고리별 수집 현황 파악 (이어쓰기 시 목표 조정)
        existing_by_cat = {cat: 0 for cat in TARGETS}
        for row in existing_rows:
            cat = row.get("category", "").strip()
            if cat in existing_by_cat:
                existing_by_cat[cat] += 1
        print(f"[collect] 기존 CSV: {len(existing_rows)}개 (uid {next_uid}부터 이어쓰기)")
        for cat, cnt in existing_by_cat.items():
            print(f"  {cat}: 기존 {cnt}개")
    else:
        existing_by_cat = {cat: 0 for cat in TARGETS}

    all_rows, uid = [], next_uid
    for category, n_target in TARGETS.items():
        risk_val  = RISK_MAP[category]
        label     = "2" if "대출" in category else "3"
        n_need    = max(0, n_target - existing_by_cat[category])
        taken     = scripts_by_cat[category][:n_need]

        if len(taken) < n_need:
            print(f"  [경고] {category}: {len(taken)}개만 확보 (추가 필요 {n_need}개)")

        for raw_text in taken:
            noised = apply_asr_noise(raw_text, noise_probs, rng)
            risks  = compute_segment_risks(noised, tokenizer, risk_val, WINDOW_SIZE, STRIDE)
            nseg   = len(risks)
            row = {
                "id":            f"short_phishing_aug_{uid:04d}",
                "text":          noised,
                "label":         label,
                "category":      category,
                "source":        "llm_short_aug",
                "filename":      "",
                "segment_risks": json.dumps(risks, ensure_ascii=False),
            }
            all_rows.append(row)
            uid += 1
            print(f"  [OK] {row['id']}  {category}  nseg={nseg}  len={len(noised)}자")

    final_rows = existing_rows + all_rows
    FIELDNAMES = ["id", "text", "label", "category", "source", "filename", "segment_risks"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n[완료] {OUT_CSV}")
    print(f"  기존 {len(existing_rows)}개 + 신규 {len(all_rows)}개 = 총 {len(final_rows)}개")
    shortfall = False
    for cat in TARGETS:
        cat_rows = [r for r in final_rows if r.get("category", "").strip() == cat]
        nsegs    = [len(json.loads(r["segment_risks"])) for r in cat_rows]
        if nsegs:
            print(f"  {cat}: {len(nsegs)}개  nseg min={min(nsegs)} avg={sum(nsegs)/len(nsegs):.1f} max={max(nsegs)}")
        if len(cat_rows) < TARGETS[cat]:
            print(f"  [경고] {cat}: 목표 {TARGETS[cat]}개 미달 ({len(cat_rows)}개 확보)")
            shortfall = True
    if shortfall:
        print("\n  → python generate_short_phishing.py --mode retry 로 부족분을 재제출하세요.")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["submit", "check", "collect", "cancel", "retry"],
                        required=True,
                        help="submit: 배치 제출 | check: 상태 확인 | collect: 결과 수집 | "
                             "cancel: 배치 취소 | retry: 부족분 재제출")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-id", default=None, help="check/collect 시 batch ID 직접 지정")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="ASR 노이즈 스케일 (0.5=절반, 1.0=원본, 기본 1.0). collect/retry 모드에 적용")
    args = parser.parse_args()

    client = OpenAI()

    if args.mode == "submit":
        submit_batch(client, args.seed)
    elif args.mode == "check":
        check_batch(client, args.batch_id)
    elif args.mode == "collect":
        collect_results(client, args.batch_id, args.seed, args.noise_scale)
    elif args.mode == "cancel":
        cancel_batch(client, args.batch_id)
    elif args.mode == "retry":
        retry_batch(client, args.seed, args.noise_scale)


if __name__ == "__main__":
    main()
