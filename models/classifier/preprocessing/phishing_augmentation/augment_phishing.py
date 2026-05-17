"""
augment_phishing.py

OpenAI Batch API를 사용하여 피싱 텍스트 증강.
토큰 한도(2M) 초과를 막기 위해 카테고리별로 분리 제출.

실행 방법:
  # 1단계: 카테고리별 배치 생성·제출
  python augment_phishing.py --mode submit --category loan
  python augment_phishing.py --mode submit --category investigation  # 첫 배치 완료 후

  # 2단계: 상태 확인
  python augment_phishing.py --mode check --category loan
  python augment_phishing.py --mode check --category investigation

  # 3단계: 결과 수집 (각각 실행 → 동일 CSV에 누적)
  python augment_phishing.py --mode collect --category loan
  python augment_phishing.py --mode collect --category investigation

  # 취소
  python augment_phishing.py --mode cancel --category loan

  # ── Top-up (기존 풀 유지 + 부족분만 추가 생성) ──────────────────────────
  # 1단계: 목표 건수를 --target으로 지정하여 제출 (기존 823건 → 825건 목표)
  python augment_phishing.py --mode submit --category loan --target 825
  python augment_phishing.py --mode submit --category investigation --target 735

  # 2단계: 상태 확인 (기존과 동일)
  python augment_phishing.py --mode check --category loan

  # 3단계: 결과 수집 시 --append 플래그로 기존 행 유지
  python augment_phishing.py --mode collect --append --category loan
  python augment_phishing.py --mode collect --append --category investigation

  # 4단계: 신규 행만 ASR 노이즈 주입 후 기존 파일에 추가
  python augment_asr_noise.py --noise-scale 0.5 --skip-existing --append

  # 5단계: 데이터셋 재빌드 (2800자 초과 원본 제외)
  python ../build_4class_dataset.py --max-orig-len 2800
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai 패키지가 필요합니다. pip install openai")
    sys.exit(1)

from pipeline_config import AUGMENTED_DIR, DEFAULT_VARIANT, TRANSCRIPTION_DIR
from prompt_loader import load_category_prompts, load_system_prompt

# ── 경로 ──────────────────────────────────────────────────────────────────
OUT_DIR = Path(AUGMENTED_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "phishing_augmented.csv"

# ── 설정 ──────────────────────────────────────────────────────────────────
MODEL                     = "gpt-5.4-mini"
TARGET_TOTAL_PER_CATEGORY = 1000
PER_CALL                  = 2
FEW_SHOT_K                = 3      # 10→3: 토큰 한도 초과 방지
SEED                      = 42

OUTPUT_FIELDS = ["id", "text", "label", "category", "source"]

# custom_id는 ASCII만 허용 → 카테고리명을 영문 키로 매핑
CATEGORY_KEY = {
    "대출 사기형":    "loan",
    "수사기관 사칭형": "investigation",
}
# --category 인자 → 한국어 카테고리명
KEY_TO_CATEGORY = {v: k for k, v in CATEGORY_KEY.items()}


def _meta_path(cat_key: str) -> Path:
    return OUT_DIR / f"phishing_batch_meta_{cat_key}.json"

def _jsonl_path(cat_key: str) -> Path:
    return OUT_DIR / f"phishing_batch_input_{cat_key}.jsonl"

def _retry_meta_path(cat_key: str) -> Path:
    return OUT_DIR / f"phishing_retry_meta_{cat_key}.json"

def _retry_jsonl_path(cat_key: str) -> Path:
    return OUT_DIR / f"phishing_retry_input_{cat_key}.jsonl"


# ── 프롬프트 ──────────────────────────────────────────────────────────────
def build_prompt(category: str, few_shots: list[str], n_generate: int, category_prompt: str) -> str:
    examples = "\n\n".join(
        f"[실제 STT 예시 {i+1}]\n{sample}" for i, sample in enumerate(few_shots)
    )
    return (
        f"{category_prompt}\n\n"
        f"[실제 STT 예시]\n{examples}\n\n"
        f"위 예시의 구어체 말투(망설임·반복·말 끊김 등 음성 전사 특성)를 따라 동일 카테고리 보이스피싱 STT 전사 텍스트를 {n_generate}개 생성하라.\n"
        "예시 문장을 직접 복사하지 말고 시나리오와 표현을 다양하게 변형하되, 말투의 자연스러움은 반드시 유지하라.\n"
        "위 예시들이 각각 다른 방식으로 통화를 시작하는 점에 주목하고, 생성 결과도 변형마다 서로 다른 시작 표현을 사용하라.\n"
        "각 결과는 반드시 [변형1], [변형2] ... 태그로 구분하라."
    )


def parse_variants(response_text: str) -> list[str]:
    parts = re.split(r"\[변형\d+\]", response_text)
    return [p.strip() for p in parts if p.strip()]


def looks_victim_initiated(text: str) -> bool:
    # 첫 3문장(문장 구분자 기준)만 검사 — 고정 길이 대신 문장 단위로 체크
    sentences = re.split(r'[.?!\n]+', text)
    head = "".join(sentences[:3]).replace(" ", "")
    victim_like_starts = [
        "제가대출문의",
        "대출문의드리려고",
        "제가신청",
        "문의드리려고전화",
        "상담받으려고전화",
        "제가먼저전화",
    ]
    return any(token in head for token in victim_like_starts)


# ── 데이터 로드 ────────────────────────────────────────────────────────────
def load_source_data(csv_path: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    counts: dict[str, int] = defaultdict(int)
    texts: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            category = (row.get("category") or "").strip()
            if category == "바로 이 목소리":
                category = "수사기관 사칭형"
            text = (row.get("text") or "").strip()
            if category:
                counts[category] += 1
            if category and 1000 <= len(text) <= 2000:
                texts[category].append(text)
    return counts, texts


def load_fewshot_from_clean_csv(csv_path: str) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            category = (row.get("category") or "").strip()
            text = (row.get("text_clean") or row.get("text") or "").strip()
            if not category or not text:
                continue
            if 1000 <= len(text) <= 2000:
                by_category[category].append(text)
    return by_category


# ── 배치 JSONL 생성 (단일 카테고리) ──────────────────────────────────────
def build_batch_jsonl(
    cat_key: str,
    category: str,
    category_prompt: str,
    system_prompt: str,
    source_texts: dict[str, list[str]],
    target_generate: int,
) -> list[dict]:
    """단일 카테고리 JSONL 생성. 요청 메타데이터 목록 반환."""
    # 카테고리별 독립 시드 → 단독 제출 시에도 재현 가능
    cat_seed = SEED + sum(ord(c) for c in category)
    rng = random.Random(cat_seed)
    meta_requests = []
    records = []

    if target_generate <= 0:
        print(f"[SKIP] {category}: 목표 충족 (추가 생성 불필요)")
        return []

    candidates = source_texts.get(category, [])
    if not candidates:
        print(f"[SKIP] few-shot 후보 없음: {category}")
        return []

    n_calls = (target_generate + PER_CALL - 1) // PER_CALL
    print(f"[INFO] {category}: 목표 {target_generate}건 → {n_calls}회 요청")

    for call_idx in range(n_calls):
        n_this_call = min(PER_CALL, target_generate - call_idx * PER_CALL)
        few_shots = rng.sample(candidates, min(FEW_SHOT_K, len(candidates)))
        prompt = build_prompt(category, few_shots, n_this_call, category_prompt)

        custom_id = f"{cat_key}_{call_idx:04d}"
        records.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 1.0,
                "max_completion_tokens": 1500,  # 4096→1500: 토큰 한도 초과 방지
            },
        })
        meta_requests.append({
            "custom_id": custom_id,
            "category":  category,
            "call_idx":  call_idx,
            "n_generate": n_this_call,
        })

    jsonl_path = _jsonl_path(cat_key)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 토큰 추정
    sample_chars = sum(len(m["content"]) for m in records[0]["body"]["messages"])
    est_tokens_per_req = sample_chars // 3 + 1500
    est_total = est_tokens_per_req * len(records)
    print(f"[batch] JSONL 생성: {jsonl_path} (총 {len(records)}개 요청)")
    print(f"[batch] 추정 토큰: 요청당 ~{est_tokens_per_req:,} × {len(records)} = ~{est_total:,} (한도 2,000,000)")
    return meta_requests


# ── 배치 제출 ──────────────────────────────────────────────────────────────
def submit_batch(client: OpenAI, cat_key: str, meta_requests: list[dict]) -> str:
    jsonl_path = _jsonl_path(cat_key)
    print("[batch] 파일 업로드 중...")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    file_id = uploaded.id
    print(f"[batch] 업로드 완료: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"phishing_augmented_{cat_key}"},
    )
    batch_id = batch.id
    print(f"[batch] 제출 완료: {batch_id}")
    print(f"[batch] 상태: {batch.status}")

    meta = {
        "batch_id":  batch_id,
        "file_id":   file_id,
        "cat_key":   cat_key,
        "status":    batch.status,
        "requests":  meta_requests,
    }
    meta_path = _meta_path(cat_key)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[batch] 메타 저장: {meta_path}")
    return batch_id


# ── 배치 상태 확인 ─────────────────────────────────────────────────────────
def check_batch(client: OpenAI, cat_key: str, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(_meta_path(cat_key).read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"[batch] 카테고리: {cat_key}")
    print(f"[batch] ID   : {batch_id}")
    print(f"[batch] 상태 : {batch.status}")
    print(f"[batch] 완료 : {counts.completed} / {counts.total}")
    print(f"[batch] 실패 : {counts.failed}")
    if batch.output_file_id:
        print(f"[batch] 출력 : {batch.output_file_id}")
    if batch.errors and batch.errors.data:
        for err in batch.errors.data:
            print(f"[batch] 오류 : [{err.code}] {err.message}")


# ── 배치 취소 ──────────────────────────────────────────────────────────────
def cancel_batch(client: OpenAI, cat_key: str, batch_id: str | None = None) -> None:
    if batch_id is None:
        meta = json.loads(_meta_path(cat_key).read_text(encoding="utf-8"))
        batch_id = meta["batch_id"]

    batch = client.batches.cancel(batch_id)
    print(f"[batch] 취소 요청: {batch_id}")
    print(f"[batch] 상태: {batch.status}")


# ── 결과 수집 ──────────────────────────────────────────────────────────────
def _parse_uid(row_id: str) -> int:
    """aug_{category}_{N} 형식에서 N 추출. 실패 시 0 반환."""
    try:
        return int(row_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def collect_results(client: OpenAI, cat_key: str, batch_id: str | None = None, append: bool = False) -> None:
    meta = json.loads(_meta_path(cat_key).read_text(encoding="utf-8"))
    if batch_id is None:
        batch_id = meta["batch_id"]

    req_map = {r["custom_id"]: r for r in meta["requests"]}

    print(f"[collect] 카테고리: {cat_key} | 배치 완료 대기 중...")
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

    # 기존 CSV 보존 전략:
    #   append=False(기본): 동일 카테고리 항목 제거 후 새 결과로 대체 (재수집 시 멱등성 보장)
    #   append=True       : 기존 행 전체 유지하고 새 결과를 누적 (top-up용)
    existing_rows: list[dict] = []
    if OUT_CSV.exists():
        with open(OUT_CSV, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if append or row.get("category", "") != KEY_TO_CATEGORY.get(cat_key, ""):
                    existing_rows.append(row)

    new_rows = []
    uid_counter: dict[str, int] = defaultdict(int)
    if append:
        for row in existing_rows:
            cat = row.get("category", "")
            uid_counter[cat] = max(uid_counter[cat], _parse_uid(row.get("id", "")))
    parse_errors: list[str] = []
    victim_filtered = 0

    for line in lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        req_info = req_map.get(custom_id, {})
        category = req_info.get("category", "unknown")
        n_generate = req_info.get("n_generate", PER_CALL)

        if result.get("error"):
            print(f"  [오류] {custom_id}: {result['error']}")
            parse_errors.append(custom_id)
            continue

        raw_text = result["response"]["body"]["choices"][0]["message"]["content"]
        variants = parse_variants(raw_text)

        if not variants:
            print(f"  [파싱 실패] {custom_id}")
            parse_errors.append(custom_id)
            continue

        for v in variants[:n_generate]:
            if looks_victim_initiated(v):
                victim_filtered += 1
                continue
            uid_counter[category] += 1
            new_rows.append({
                "id":       f"aug_{category}_{uid_counter[category]:04d}",
                "text":     v,
                "label":    1,
                "category": category,
                "source":   "augmented_llm",
            })

    all_rows = existing_rows + new_rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[collect] 완료: {len(new_rows)}건 추가 → {OUT_CSV} (전체 {len(all_rows)}건)")
    if victim_filtered:
        print(f"[collect] 발화 주체 필터링: {victim_filtered}건 제거")
    if parse_errors:
        print(f"[collect] 파싱 실패: {len(parse_errors)}건")
    dist = Counter(r["category"] for r in new_rows)
    for cat, cnt in dist.items():
        print(f"  {cat}: {cnt}건")


# ── retry: 실패 call_idx 탐색 ─────────────────────────────────────────────
def find_missing_calls(client: OpenAI, cat_key: str) -> list[int]:
    """원본 + 기존 retry 배치 출력을 모두 분석해 여전히 부족한 call_idx 목록 반환."""
    meta = json.loads(_meta_path(cat_key).read_text(encoding="utf-8"))
    orig_req_map = {r["custom_id"]: r for r in meta["requests"]}

    def check_output(output_file_id: str, req_map: dict) -> set[int]:
        """출력 파일에서 필터 통과 변형이 충분한 call_idx 집합 반환."""
        content = client.files.content(output_file_id).content
        lines = content.decode("utf-8").strip().split("\n")
        ok: set[int] = set()
        for line in lines:
            result = json.loads(line)
            custom_id = result["custom_id"]
            if custom_id not in req_map or result.get("error"):
                continue
            raw = result["response"]["body"]["choices"][0]["message"]["content"]
            variants = parse_variants(raw)
            n_generate = req_map[custom_id]["n_generate"]
            valid = [v for v in variants if not looks_victim_initiated(v)]
            if len(valid) >= n_generate:
                ok.add(req_map[custom_id]["call_idx"])
        return ok

    # 원본 배치 체크
    orig_batch = client.batches.retrieve(meta["batch_id"])
    if not orig_batch.output_file_id:
        raise SystemExit(f"[retry] 원본 출력 파일 없음 — 배치 상태: {orig_batch.status}")
    succeeded = check_output(orig_batch.output_file_id, orig_req_map)

    # 기존 retry 배치 전체 history 체크
    retry_meta_path = _retry_meta_path(cat_key)
    if retry_meta_path.exists():
        history = json.loads(retry_meta_path.read_text(encoding="utf-8"))
        if isinstance(history, dict):
            history = [history]
        for entry in history:
            retry_req_map = {r["custom_id"]: r for r in entry["requests"]}
            retry_batch = client.batches.retrieve(entry["batch_id"])
            if retry_batch.output_file_id:
                succeeded |= check_output(retry_batch.output_file_id, retry_req_map)

    all_call_idxs = [r["call_idx"] for r in meta["requests"]]
    missing = [i for i in all_call_idxs if i not in succeeded]
    print(f"[retry] {cat_key}: 전체 {len(all_call_idxs)}건 중 실패/불완전 {len(missing)}건")
    if missing:
        print(f"  call_idx: {missing}")
    return missing


# ── retry: JSONL 생성 ──────────────────────────────────────────────────────
def build_retry_jsonl(
    cat_key: str,
    category: str,
    category_prompt: str,
    system_prompt: str,
    source_texts: dict[str, list[str]],
    missing_calls: list[int],
) -> list[dict]:
    """원본과 동일한 rng 시퀀스를 재생해 실패 call_idx만 JSONL 저장."""
    meta = json.loads(_meta_path(cat_key).read_text(encoding="utf-8"))
    req_map = {r["call_idx"]: r for r in meta["requests"]}
    missing_set = set(missing_calls)

    cat_seed = SEED + sum(ord(c) for c in category)
    rng = random.Random(cat_seed)

    n_calls = max(req_map.keys()) + 1
    candidates = source_texts.get(category, [])
    retry_requests = []
    records = []

    for call_idx in range(n_calls):
        # rng 시퀀스 반드시 동일하게 소비
        few_shots = rng.sample(candidates, min(FEW_SHOT_K, len(candidates)))
        if call_idx not in missing_set:
            continue
        n_this_call = req_map[call_idx]["n_generate"]
        prompt = build_prompt(category, few_shots, n_this_call, category_prompt)
        custom_id = f"{cat_key}_retry_{call_idx:04d}"
        records.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 1.0,
                "max_completion_tokens": 1500,
            },
        })
        retry_requests.append({
            "custom_id":  custom_id,
            "category":   category,
            "call_idx":   call_idx,
            "n_generate": n_this_call,
        })

    jsonl_path = _retry_jsonl_path(cat_key)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[retry] JSONL 생성: {jsonl_path} ({len(records)}개 요청)")
    return retry_requests


# ── retry: 배치 제출 ───────────────────────────────────────────────────────
def submit_retry_batch(client: OpenAI, cat_key: str, retry_requests: list[dict]) -> str:
    jsonl_path = _retry_jsonl_path(cat_key)
    print("[retry] 파일 업로드 중...")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    file_id = uploaded.id
    print(f"[retry] 업로드 완료: {file_id}")

    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"phishing_retry_{cat_key}"},
    )
    batch_id = batch.id
    print(f"[retry] 제출 완료: {batch_id} | 상태: {batch.status}")

    new_entry = {
        "batch_id": batch_id,
        "file_id":  file_id,
        "cat_key":  cat_key,
        "status":   batch.status,
        "requests": retry_requests,
    }
    # 기존 history에 누적 (덮어쓰지 않음)
    retry_meta_path = _retry_meta_path(cat_key)
    history = []
    if retry_meta_path.exists():
        existing = json.loads(retry_meta_path.read_text(encoding="utf-8"))
        history = existing if isinstance(existing, list) else [existing]
    history.append(new_entry)
    retry_meta_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[retry] 메타 저장: {retry_meta_path} (누적 {len(history)}회차)")
    return batch_id


# ── retry: 결과 수집·병합 ──────────────────────────────────────────────────
def collect_retry_results(client: OpenAI, cat_key: str, batch_id: str | None = None) -> None:
    history = json.loads(_retry_meta_path(cat_key).read_text(encoding="utf-8"))
    if isinstance(history, dict):
        history = [history]

    # 최신 배치 완료 대기
    latest = history[-1]
    wait_id = batch_id or latest["batch_id"]
    category = KEY_TO_CATEGORY[cat_key]

    print(f"[retry] 카테고리: {cat_key} | 최신 배치 완료 대기 중...")
    while True:
        batch = client.batches.retrieve(wait_id)
        print(f"  상태: {batch.status} | {batch.request_counts.completed}/{batch.request_counts.total}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(30)

    if batch.status != "completed":
        print(f"[retry] 배치 실패: {batch.status}")
        return

    # 전체 history에서 retried call_idx 수집 + 각 배치 출력 다운로드
    retried_call_idxs: set[int] = set()
    all_retry_lines: list[tuple[str, dict]] = []  # (line, req_map)
    for entry in history:
        req_map = {r["custom_id"]: r for r in entry["requests"]}
        retried_call_idxs |= {r["call_idx"] for r in entry["requests"]}
        b = client.batches.retrieve(entry["batch_id"])
        if b.output_file_id:
            content = client.files.content(b.output_file_id).content
            for line in content.decode("utf-8").strip().split("\n"):
                all_retry_lines.append((line, req_map))

    # 원본 배치 출력 재다운로드 (retry 대상 call_idx 제외하고 처리)
    orig_meta = json.loads(_meta_path(cat_key).read_text(encoding="utf-8"))
    orig_req_map = {r["custom_id"]: r for r in orig_meta["requests"]}
    orig_batch = client.batches.retrieve(orig_meta["batch_id"])
    orig_content = client.files.content(orig_batch.output_file_id).content
    orig_lines = orig_content.decode("utf-8").strip().split("\n")

    # 다른 카테고리 행만 유지
    other_rows: list[dict] = []
    if OUT_CSV.exists():
        with open(OUT_CSV, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("category") != category:
                    other_rows.append(row)

    def process_lines(lines, req_map, skip_call_idxs=None):
        rows, errors, filtered = [], [], 0
        for line in lines:
            result = json.loads(line)
            custom_id = result["custom_id"]
            req_info = req_map.get(custom_id, {})
            if skip_call_idxs and req_info.get("call_idx") in skip_call_idxs:
                continue
            n_generate = req_info.get("n_generate", PER_CALL)
            if result.get("error"):
                errors.append(custom_id)
                continue
            raw = result["response"]["body"]["choices"][0]["message"]["content"]
            variants = parse_variants(raw)
            if not variants:
                errors.append(custom_id)
                continue
            for v in variants[:n_generate]:
                if looks_victim_initiated(v):
                    filtered += 1
                    continue
                rows.append(v)
        return rows, errors, filtered

    orig_texts, orig_errors, orig_filtered = process_lines(orig_lines, orig_req_map, skip_call_idxs=retried_call_idxs)
    retry_texts, retry_errors, retry_filtered = [], [], 0
    for line, req_map in all_retry_lines:
        t, e, f = process_lines([line], req_map)
        retry_texts += t
        retry_errors += e
        retry_filtered += f

    new_rows = []
    parse_errors = orig_errors + retry_errors
    victim_filtered = orig_filtered + retry_filtered
    for i, text in enumerate(orig_texts + retry_texts, start=1):
        new_rows.append({
            "id":       f"aug_{category}_{i:04d}",
            "text":     text,
            "label":    1,
            "category": category,
            "source":   "augmented_llm",
        })

    all_rows = other_rows + new_rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[retry] 완료: {len(new_rows)}건 추가 → {OUT_CSV} (전체 {len(all_rows)}건)")
    if victim_filtered:
        print(f"[retry] 발화 주체 필터링: {victim_filtered}건 제거")
    if parse_errors:
        print(f"[retry] 파싱 실패: {len(parse_errors)}건")


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API 기반 피싱 텍스트 증강 (카테고리별 분리 제출)")
    parser.add_argument(
        "--mode",
        choices=["submit", "check", "collect", "cancel",
                 "submit-retry", "check-retry", "collect-retry"],
        required=True,
        help="submit/check/collect/cancel | submit-retry: 실패 재제출 | check-retry: 재시도 확인 | collect-retry: 재시도 결과 병합",
    )
    parser.add_argument(
        "--category",
        choices=["loan", "investigation"],
        required=True,
        help="loan: 대출 사기형 | investigation: 수사기관 사칭형",
    )
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--fewshot-csv", default="", help="정제된 few-shot CSV (비우면 원본 STT 사용)")
    parser.add_argument(
        "--target", type=int, default=None,
        help="LLM 증강 목표 총 건수 (생략 시 TARGET_TOTAL_PER_CATEGORY 사용). "
             "top-up 시: 현재 CSV 기존 건수를 뺀 만큼만 생성.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="collect 시 기존 동일 카테고리 행을 유지하고 새 결과를 추가 (top-up용).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    client = OpenAI(api_key=api_key)
    cat_key  = args.category
    category = KEY_TO_CATEGORY[cat_key]

    if args.mode == "submit":
        system_prompt    = load_system_prompt()
        category_prompts = load_category_prompts()

        if category not in category_prompts:
            raise SystemExit(f"[ERROR] 카테고리 프롬프트 없음: {category}")

        source_csv = os.path.join(TRANSCRIPTION_DIR, DEFAULT_VARIANT, "phishing.csv")
        _, source_texts = load_source_data(source_csv)

        if args.fewshot_csv:
            source_texts = load_fewshot_from_clean_csv(args.fewshot_csv)
            print(f"[INFO] 정제 few-shot 사용: {args.fewshot_csv}")
        else:
            print(f"[INFO] 원본 STT few-shot 사용: {source_csv}")

        # 기존 CSV에서 현재 카테고리 건수 파악
        existing_count = 0
        if OUT_CSV.exists():
            with open(OUT_CSV, "r", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if (row.get("category") or "").strip() == category:
                        existing_count += 1

        llm_target = args.target if args.target is not None else TARGET_TOTAL_PER_CATEGORY
        target_generate = max(llm_target - existing_count, 0)
        print(f"[INFO] {category}: 기존 {existing_count}건, 목표 {llm_target}건 → 추가 생성 {target_generate}건")

        meta_requests = build_batch_jsonl(
            cat_key, category, category_prompts[category],
            system_prompt, source_texts, target_generate,
        )
        if meta_requests:
            submit_batch(client, cat_key, meta_requests)

    elif args.mode == "check":
        check_batch(client, cat_key, args.batch_id)

    elif args.mode == "cancel":
        cancel_batch(client, cat_key, args.batch_id)

    elif args.mode == "collect":
        collect_results(client, cat_key, args.batch_id, append=args.append)

    elif args.mode == "submit-retry":
        system_prompt    = load_system_prompt()
        category_prompts = load_category_prompts()
        source_csv = os.path.join(TRANSCRIPTION_DIR, DEFAULT_VARIANT, "phishing.csv")
        _, source_texts = load_source_data(source_csv)
        if args.fewshot_csv:
            source_texts = load_fewshot_from_clean_csv(args.fewshot_csv)

        missing = find_missing_calls(client, cat_key)
        if not missing:
            print("[retry] 실패 요청 없음 — retry 불필요")
            return
        retry_requests = build_retry_jsonl(
            cat_key, category, category_prompts[category],
            system_prompt, source_texts, missing,
        )
        submit_retry_batch(client, cat_key, retry_requests)

    elif args.mode == "check-retry":
        history = json.loads(_retry_meta_path(cat_key).read_text(encoding="utf-8"))
        if isinstance(history, dict):
            history = [history]
        latest_id = args.batch_id or history[-1]["batch_id"]
        check_batch(client, cat_key, latest_id)

    elif args.mode == "collect-retry":
        collect_retry_results(client, cat_key, args.batch_id)


if __name__ == "__main__":
    main()
