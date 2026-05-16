"""
preprocess_augmented_data.py

LLM 생성 증강 데이터 전처리 파이프라인

[정상 데이터 - long_normal_augmented.csv]
  ASR 노이즈 및 segment_risks(all=1)는 생성(collect) 시 이미 적용됨.
  문장부호 제거(clean_text)만 추가 수행.
  → output/long_normal_preprocessed.csv

[피싱 데이터 - short_phishing_augmented.csv]
  순서: 위험도 라벨링 → 세그먼트 위험도 계산 → 문장부호 제거
  (ASR 노이즈는 생성 시 이미 적용됨)
  → output/short_phishing_preprocessed.csv

모드:
  --mode clean-normal     : 정상 데이터 문장부호 제거 (즉시 완료, API 불필요)
  --mode submit-phishing  : 피싱 데이터 위험도 라벨링 Batch API 제출
  --mode check-phishing   : 배치 상태 확인
  --mode collect-phishing : 결과 수집 → 세그먼트 위험도 계산 → CSV 저장
  --mode cancel-phishing  : 배치 취소

사용 예:
  python preprocess_augmented_data.py --mode clean-normal
  python preprocess_augmented_data.py --mode submit-phishing
  python preprocess_augmented_data.py --mode check-phishing
  python preprocess_augmented_data.py --mode collect-phishing [--poll]
  python preprocess_augmented_data.py --mode cancel-phishing
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # optional for clean-normal mode

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
OUT_DIR     = BASE_DIR / "output"
STN_DIR     = BASE_DIR / "stn_labeling"
PROMPT_FILE = STN_DIR / "prompts" / "system_prompt.txt"

IN_NORMAL_CSV   = OUT_DIR / "long_normal_augmented.csv"
OUT_NORMAL_CSV  = OUT_DIR / "long_normal_preprocessed.csv"

IN_PHISHING_CSV  = OUT_DIR / "short_phishing_augmented.csv"
OUT_PHISHING_CSV = OUT_DIR / "short_phishing_preprocessed.csv"
PHISHING_BATCH_META = OUT_DIR / "phishing_label_batch_meta.json"
PHISHING_BATCH_JSONL = OUT_DIR / "phishing_label_batch_input.jsonl"

MODEL          = "gpt-4o-mini"
TOKENIZER_NAME = "klue/roberta-base"
WINDOW_SIZE    = 64
STRIDE         = 32

# 4클래스 레이블 기준으로 피싱으로 처리할 레이블
PHISHING_LABELS = {"2", "3"}

CSV_COLUMNS = ["id", "text", "label", "category", "source", "filename", "segment_risks"]

# ── 유틸 ─────────────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")


def clean_text(text: str) -> str:
    """문장부호 제거 — 한글·영문·숫자·공백만 유지."""
    cleaned = _PUNCT_RE.sub("", text)
    return re.sub(r" +", " ", cleaned).strip()


def load_system_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"시스템 프롬프트 없음: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


# ── segment_risks 계산 (피싱: 문장별 위험도 기반) ─────────────────────────────
def compute_phishing_segment_risks(
    sentences: list[dict],
    tokenizer,
    window_size: int,
    stride: int,
) -> list[int]:
    """
    sentences: [{"sentence": str, "risk": int}, ...]
    clean_text를 적용한 뒤 슬라이딩 윈도우로 세그먼트 위험도 계산.
    """
    SEP = " "
    full_text = ""
    char_spans: list[tuple[int, int, int]] = []  # (start, end, risk)

    for s in sentences:
        cleaned_s = clean_text(s["sentence"])
        if not cleaned_s:
            continue
        start = len(full_text)
        full_text += cleaned_s
        end = len(full_text)
        char_spans.append((start, end, int(s["risk"])))
        full_text += SEP

    if not full_text.strip():
        return []

    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    tok_strs: list[str] = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    n = len(offsets)
    if n == 0:
        return []

    # 토큰별 위험도: 문자 오프셋 기반 매핑
    token_risks: list[int] = []
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            token_risks.append(1)
            continue
        risk = 1
        for sent_start, sent_end, sent_risk in char_spans:
            if tok_start < sent_end and tok_end > sent_start:
                risk = sent_risk
                break
        token_risks.append(risk)

    # 슬라이딩 윈도우 적용 (윈도우 내 최대 위험도)
    segment_risks: list[int] = []
    pos = 0
    while pos < n:
        end = min(pos + window_size, n) - 1
        while end + 1 < n and tok_strs[end + 1].startswith("##"):
            end += 1
        segment_risks.append(max(token_risks[pos: end + 1]))
        if end + 1 >= n:
            break
        next_pos = pos + stride
        while next_pos < n and tok_strs[next_pos].startswith("##"):
            next_pos += 1
        if next_pos >= n:
            break
        pos = next_pos

    return segment_risks


# ── segment_risks 계산 (정상: all=1) ─────────────────────────────────────────
def compute_normal_segment_risks(text: str, tokenizer, window_size: int, stride: int) -> list[int]:
    """clean_text 적용 후 슬라이딩 윈도우, 모두 위험도=1."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    enc = tokenizer(
        cleaned,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    n = len(enc["offset_mapping"])
    if n == 0:
        return []
    tok_strs = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    risks: list[int] = []
    pos = 0
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


# ── 응답 파싱/검증 ────────────────────────────────────────────────────────────
def parse_response(raw: str) -> list[dict] | None:
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate_sentences(sentences: list[dict]) -> bool:
    if not sentences:
        return False
    for s in sentences:
        if not isinstance(s, dict):
            return False
        if "sentence" not in s or "risk" not in s:
            return False
        if int(s["risk"]) not in (1, 2, 3):
            return False
        if not isinstance(s["sentence"], str) or not s["sentence"].strip():
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 모드 1: 정상 데이터 문장부호 제거
# ══════════════════════════════════════════════════════════════════════════════
def mode_clean_normal() -> None:
    """
    long_normal_augmented.csv의 text 필드에 clean_text() 적용.
    segment_risks는 기존 값을 그대로 유지 (생성 시 이미 clean_text 기반으로 계산됨).

    ⚠ 주의: generate_long_normal.py의 collect 단계에서 Track A (상담 대화)에도
       label="0", category="일상 대화"가 잘못 기록된 버그가 있음.
       Track A (uid 0001~0048, n_target=95 기준)는 label=0, category="상담 대화"이어야 함.
       이 스크립트는 그 버그를 그대로 유지하고 text 필드만 정리함.
       label/category 수정이 필요하면 --fix-labels 옵션을 추가해 직접 수정 가능.
    """
    if not IN_NORMAL_CSV.exists():
        raise FileNotFoundError(f"입력 파일 없음: {IN_NORMAL_CSV}")

    rows = []
    with IN_NORMAL_CSV.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"[ERROR] 입력 파일이 비어 있습니다: {IN_NORMAL_CSV}")

    print(f"[clean-normal] {len(rows)}개 로드")
    label_fix = 0
    out_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["text"] = clean_text(row.get("text", ""))
        # label 버그 수정: 일상 대화는 label=1
        if new_row.get("category") == "일상 대화" and new_row.get("label") == "0":
            new_row["label"] = "1"
            label_fix += 1
        out_rows.append(new_row)
    if label_fix:
        print(f"[clean-normal] label 수정: 일상 대화 {label_fix}행 (0 → 1)")

    OUT_NORMAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_NORMAL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[clean-normal] 완료: {OUT_NORMAL_CSV}  ({len(out_rows)}개)")
    print("  ※ segment_risks는 기존 값 유지 (all=1, 슬라이딩 윈도우 기반)")


# ══════════════════════════════════════════════════════════════════════════════
# 모드 2: 피싱 데이터 위험도 라벨링 Batch API 제출
# ══════════════════════════════════════════════════════════════════════════════
def mode_submit_phishing(client: OpenAI) -> None:
    """
    short_phishing_augmented.csv의 각 대화를 Batch API로 문장별 위험도 라벨링 요청.
    label=2 (대출 사기형), label=3 (수사기관 사칭형)을 피싱으로 처리.
    """
    if not IN_PHISHING_CSV.exists():
        raise FileNotFoundError(f"입력 파일 없음: {IN_PHISHING_CSV}")

    system_prompt = load_system_prompt()

    rows = []
    with IN_PHISHING_CSV.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid  = (row.get("id") or "").strip()
            text = (row.get("text") or "").strip()
            lbl  = str(row.get("label", "")).strip()
            if rid and text and lbl in PHISHING_LABELS:
                rows.append(row)

    if not rows:
        raise SystemExit("[ERROR] 처리할 피싱 행이 없습니다.")

    print(f"[submit-phishing] 피싱 대화 {len(rows)}건 — Batch API 제출 준비")

    # 기존 완료 파일 확인 (재제출 방지)
    done_ids: set[str] = set()
    if OUT_PHISHING_CSV.exists():
        with OUT_PHISHING_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done_ids.add(row.get("id", ""))
        if done_ids:
            rows = [r for r in rows if (r.get("id") or "").strip() not in done_ids]
            print(f"[submit-phishing] 이미 완료 {len(done_ids)}건 제외 → 나머지 {len(rows)}건 제출")

    if not rows:
        print("[submit-phishing] 모든 대화가 이미 처리됨. 제출 불필요.")
        return

    # JSONL 생성
    row_meta: dict[str, dict] = {}
    PROMPT_PREFIX = "다음 보이스피싱 STT 텍스트를 발화 단위로 분리하고 각 단위에 위험도를 부여하라.\n\n[STT 텍스트]\n"

    with PHISHING_BATCH_JSONL.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            row_id = (row.get("id") or "").strip()
            text   = (row.get("text") or "").strip()
            custom_id = f"phishing_label_{idx:06d}"
            row_meta[custom_id] = {
                "id":       row_id,
                "label":    str(row.get("label", "2")).strip(),
                "category": (row.get("category") or "").strip(),
                "source":   (row.get("source") or "").strip(),
                "filename": (row.get("filename") or "").strip(),
            }
            request = {
                "custom_id": custom_id,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": PROMPT_PREFIX + text},
                    ],
                    "max_tokens": 8192,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"[submit-phishing] JSONL 생성: {len(rows)}건  →  {PHISHING_BATCH_JSONL}")

    # 파일 업로드 및 배치 제출
    with PHISHING_BATCH_JSONL.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"[submit-phishing] 파일 업로드: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "phishing_aug_sentence_labeling"},
    )
    print(f"[submit-phishing] 배치 제출 완료: {batch.id}  상태: {batch.status}")

    meta = {
        "batch_id":    batch.id,
        "file_id":     uploaded.id,
        "request_cnt": len(rows),
        "status":      batch.status,
        "row_meta":    row_meta,
    }
    PHISHING_BATCH_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[submit-phishing] 메타 저장: {PHISHING_BATCH_META}")
    print(f"\n다음 명령어로 상태 확인:")
    print(f"  python preprocess_augmented_data.py --mode check-phishing")
    print(f"완료 후 결과 수집:")
    print(f"  python preprocess_augmented_data.py --mode collect-phishing [--poll]")


# ══════════════════════════════════════════════════════════════════════════════
# 모드 3: 배치 상태 확인
# ══════════════════════════════════════════════════════════════════════════════
def mode_check_phishing(client: OpenAI) -> None:
    if not PHISHING_BATCH_META.exists():
        raise SystemExit(f"[ERROR] 메타 파일 없음: {PHISHING_BATCH_META}\n먼저 --mode submit-phishing 실행")
    meta     = json.loads(PHISHING_BATCH_META.read_text(encoding="utf-8"))
    batch_id = meta["batch_id"]
    batch    = client.batches.retrieve(batch_id)
    counts   = batch.request_counts
    print(f"[check-phishing] batch_id : {batch_id}")
    print(f"[check-phishing] 상태     : {batch.status}")
    print(f"[check-phishing] 완료/전체: {counts.completed} / {counts.total}  실패: {counts.failed}")
    if batch.output_file_id:
        print(f"[check-phishing] 출력 파일: {batch.output_file_id}")


# ══════════════════════════════════════════════════════════════════════════════
# 모드 4: 배치 취소
# ══════════════════════════════════════════════════════════════════════════════
def mode_cancel_phishing(client: OpenAI) -> None:
    if not PHISHING_BATCH_META.exists():
        raise SystemExit(f"[ERROR] 메타 파일 없음: {PHISHING_BATCH_META}")
    meta     = json.loads(PHISHING_BATCH_META.read_text(encoding="utf-8"))
    batch_id = meta["batch_id"]
    batch    = client.batches.cancel(batch_id)
    print(f"[cancel-phishing] 취소 요청: {batch_id}  상태: {batch.status}")


# ══════════════════════════════════════════════════════════════════════════════
# 모드 5: 피싱 결과 수집 → segment_risks 재계산 → clean_text → CSV 저장
# ══════════════════════════════════════════════════════════════════════════════
def mode_collect_phishing(client: OpenAI, poll: bool = False, poll_interval: int = 60) -> None:
    """
    처리 순서:
      1. 위험도 라벨링 결과 수집 (Batch API)
      2. 세그먼트 위험도 계산 (슬라이딩 윈도우 + 문장별 위험도)
      3. 문장부호 제거 (clean_text → text 필드)
    """
    if not PHISHING_BATCH_META.exists():
        raise SystemExit(f"[ERROR] 메타 파일 없음: {PHISHING_BATCH_META}\n먼저 --mode submit-phishing 실행")

    meta     = json.loads(PHISHING_BATCH_META.read_text(encoding="utf-8"))
    batch_id = meta["batch_id"]
    row_meta = meta["row_meta"]  # custom_id → {id, label, category, source, filename}

    # 원본 text 매핑 (노이즈 적용된 원본 텍스트를 clean_text에 사용)
    orig_text_map: dict[str, str] = {}
    if IN_PHISHING_CSV.exists():
        with IN_PHISHING_CSV.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                orig_text_map[(row.get("id") or "").strip()] = (row.get("text") or "").strip()

    # 배치 완료 대기
    print(f"[collect-phishing] 배치 상태 확인: {batch_id}")
    while True:
        batch  = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  상태: {batch.status}  완료: {counts.completed}/{counts.total}  실패: {counts.failed}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        if not poll:
            print("[collect-phishing] 아직 처리 중. --poll 옵션으로 자동 대기 가능.")
            return
        print(f"  {poll_interval}초 후 재확인...")
        time.sleep(poll_interval)

    # 출력 파일 확인
    if batch.status != "completed":
        if batch.output_file_id:
            print(f"[collect-phishing] 배치 상태: {batch.status} — 부분 결과 수집 시도")
        else:
            print(f"[collect-phishing] 배치 실패: {batch.status}. 수집 가능한 출력 없음.")
            print("[collect-phishing] → --mode submit-phishing 으로 재제출하세요.")
            return

    # 결과 다운로드
    result_text = client.files.content(batch.output_file_id).text
    print(f"[collect-phishing] 결과 다운로드 완료")

    # 오류 파일 출력
    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text
        for line in error_text.strip().split("\n"):
            if line.strip():
                err = json.loads(line)
                cid = err.get("custom_id", "?")
                m   = row_meta.get(cid)
                rid = m["id"] if m else cid
                print(f"  [오류] {rid}: {err.get('error', err)}")

    # 토크나이저 로드
    try:
        from transformers import AutoTokenizer
        print(f"[collect-phishing] 토크나이저 로드: {TOKENIZER_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except ImportError:
        raise SystemExit("[ERROR] pip install transformers")

    # 기존 출력 파일 확인 (중복 제거)
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if OUT_PHISHING_CSV.exists():
        with OUT_PHISHING_CSV.open("r", encoding="utf-8-sig") as f:
            existing_rows = list(csv.DictReader(f))
            done_ids = {r.get("id", "") for r in existing_rows}
        print(f"[collect-phishing] 기존 완료: {len(done_ids)}건 → 이어쓰기")

    new_rows: list[dict] = []
    fail_count = 0

    for line in result_text.strip().split("\n"):
        if not line.strip():
            continue
        result    = json.loads(line)
        custom_id = result.get("custom_id", "")
        m         = row_meta.get(custom_id)
        if not m:
            print(f"  [경고] 매핑 없음: {custom_id}")
            fail_count += 1
            continue

        row_id = m["id"]
        if row_id in done_ids:
            continue

        if result.get("error"):
            print(f"  [오류] {row_id}: {result['error']}")
            fail_count += 1
            continue

        try:
            raw = result["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            print(f"  [오류] {row_id} 응답 구조 오류: {e}")
            fail_count += 1
            continue

        sentences = parse_response(raw)
        if not sentences or not validate_sentences(sentences):
            print(f"  [경고] {row_id} 파싱/검증 실패. 균일 위험도로 폴백.")
            # 폴백: category 기준으로 균일 위험도 사용
            fallback_risk = 3 if str(m.get("label", "2")) == "3" else 2
            orig_text = orig_text_map.get(row_id, "")
            sentences = [{"sentence": orig_text, "risk": fallback_risk}]
            fail_count += 1

        # ── 순서 1: 위험도 라벨 기반 세그먼트 위험도 계산 ─────────────────
        seg_risks = compute_phishing_segment_risks(sentences, tokenizer, WINDOW_SIZE, STRIDE)

        # ── 순서 2: ASR 노이즈는 이미 원본에 적용됨 (skip) ────────────────

        # ── 순서 3: 문장부호 제거 ─────────────────────────────────────────
        orig_text  = orig_text_map.get(row_id, "")
        final_text = clean_text(orig_text)

        risk_counts = {1: 0, 2: 0, 3: 0}
        for s in sentences:
            risk_counts[int(s["risk"])] += 1

        new_rows.append({
            "id":            row_id,
            "text":          final_text,
            "label":         m["label"],
            "category":      m["category"],
            "source":        m["source"],
            "filename":      m["filename"],
            "segment_risks": json.dumps(seg_risks, ensure_ascii=False),
        })
        done_ids.add(row_id)
        print(
            f"  [OK] {row_id} ({m['category']})  nseg={len(seg_risks)}"
            f"  위험도1:{risk_counts[1]} 2:{risk_counts[2]} 3:{risk_counts[3]}"
        )

    # CSV 저장
    final_rows = existing_rows + new_rows
    OUT_PHISHING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PHISHING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n[collect-phishing] 완료: {OUT_PHISHING_CSV}")
    print(f"  기존 {len(existing_rows)}개 + 신규 {len(new_rows)}개 = 총 {len(final_rows)}개")
    if fail_count > 0:
        print(f"  [경고] 파싱 실패/오류: {fail_count}건 (균일 위험도 폴백 또는 제외)")
        print(f"  재제출이 필요하면: python preprocess_augmented_data.py --mode submit-phishing")


# ══════════════════════════════════════════════════════════════════════════════
# 모드 6: 피싱 동기 처리 (Batch API 없이 즉시 실행)
# ══════════════════════════════════════════════════════════════════════════════
def mode_sync_phishing(client: OpenAI) -> None:
    """
    Batch API 대신 동기 API로 즉시 처리.
    처리 순서:
      1. 위험도 라벨링 (chat.completions.create)
      2. 세그먼트 위험도 계산 (슬라이딩 윈도우 + 문장별 위험도)
      3. 문장부호 제거 (clean_text → text 필드)
    """
    if not IN_PHISHING_CSV.exists():
        raise FileNotFoundError(f"입력 파일 없음: {IN_PHISHING_CSV}")

    system_prompt = load_system_prompt()
    PROMPT_PREFIX = "다음 보이스피싱 STT 텍스트를 발화 단위로 분리하고 각 단위에 위험도를 부여하라.\n\n[STT 텍스트]\n"

    rows = []
    with IN_PHISHING_CSV.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lbl  = str(row.get("label", "")).strip()
            text = (row.get("text") or "").strip()
            if text and lbl in PHISHING_LABELS:
                rows.append(row)

    if not rows:
        raise SystemExit("[ERROR] 처리할 피싱 행이 없습니다.")

    # 기존 완료 행 제외
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if OUT_PHISHING_CSV.exists():
        with OUT_PHISHING_CSV.open("r", encoding="utf-8-sig") as f:
            existing_rows = list(csv.DictReader(f))
            done_ids = {r.get("id", "") for r in existing_rows}
        rows = [r for r in rows if (r.get("id") or "").strip() not in done_ids]
        print(f"[sync-phishing] 기존 완료 {len(done_ids)}건 제외 → 나머지 {len(rows)}건 처리")

    if not rows:
        print("[sync-phishing] 모든 행이 이미 처리됨.")
        return

    print(f"[sync-phishing] {len(rows)}건 동기 처리 시작")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except ImportError:
        raise SystemExit("[ERROR] pip install transformers")

    new_rows: list[dict] = []
    fail_count = 0

    for i, row in enumerate(rows):
        row_id = (row.get("id") or "").strip()
        text   = (row.get("text") or "").strip()
        lbl    = str(row.get("label", "2")).strip()

        print(f"  [{i+1}/{len(rows)}] {row_id} ...", end=" ", flush=True)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": PROMPT_PREFIX + text},
                ],
                max_tokens=8192,
            )
            raw = resp.choices[0].message.content
            sentences = parse_response(raw)
        except Exception as e:
            print(f"오류: {e}")
            sentences = None
            fail_count += 1

        if not sentences or not validate_sentences(sentences):
            fallback_risk = 3 if lbl == "3" else 2
            sentences = [{"sentence": text, "risk": fallback_risk}]
            if not (sentences and validate_sentences(sentences)):
                fail_count += 1

        # 1. 세그먼트 위험도 계산
        seg_risks = compute_phishing_segment_risks(sentences, tokenizer, WINDOW_SIZE, STRIDE)

        # 2. ASR 노이즈 이미 적용됨 (skip)

        # 3. 문장부호 제거
        final_text = clean_text(text)

        risk_counts = {1: 0, 2: 0, 3: 0}
        for s in sentences:
            risk_counts[int(s["risk"])] += 1

        print(f"nseg={len(seg_risks)}  위험도1:{risk_counts[1]} 2:{risk_counts[2]} 3:{risk_counts[3]}")

        new_rows.append({
            "id":            row_id,
            "text":          final_text,
            "label":         lbl,
            "category":      (row.get("category") or "").strip(),
            "source":        (row.get("source") or "").strip(),
            "filename":      (row.get("filename") or "").strip(),
            "segment_risks": json.dumps(seg_risks, ensure_ascii=False),
        })

    final_rows = existing_rows + new_rows
    OUT_PHISHING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PHISHING_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n[sync-phishing] 완료: {OUT_PHISHING_CSV}")
    print(f"  기존 {len(existing_rows)}개 + 신규 {len(new_rows)}개 = 총 {len(final_rows)}개")
    if fail_count:
        print(f"  [경고] 폴백 처리: {fail_count}건")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 생성 증강 데이터 전처리")
    parser.add_argument(
        "--mode",
        choices=["clean-normal", "submit-phishing", "check-phishing",
                 "collect-phishing", "cancel-phishing", "sync-phishing"],
        required=True,
        help=(
            "clean-normal: 정상 데이터 문장부호 제거 | "
            "submit-phishing: 피싱 위험도 라벨링 배치 제출 | "
            "check-phishing: 배치 상태 확인 | "
            "collect-phishing: 배치 결과 수집 및 저장 | "
            "cancel-phishing: 배치 취소 | "
            "sync-phishing: 동기 API로 즉시 처리 (배치 불필요)"
        ),
    )
    parser.add_argument(
        "--poll", action="store_true",
        help="collect-phishing 시 완료될 때까지 자동 폴링",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="폴링 간격 (초, 기본 60)",
    )
    args = parser.parse_args()

    if args.mode == "clean-normal":
        mode_clean_normal()
        return

    # API가 필요한 모드
    if OpenAI is None:
        raise SystemExit("[ERROR] pip install openai")
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")
    client = OpenAI(api_key=api_key)

    if args.mode == "submit-phishing":
        mode_submit_phishing(client)
    elif args.mode == "check-phishing":
        mode_check_phishing(client)
    elif args.mode == "cancel-phishing":
        mode_cancel_phishing(client)
    elif args.mode == "collect-phishing":
        mode_collect_phishing(client, poll=args.poll, poll_interval=args.poll_interval)
    elif args.mode == "sync-phishing":
        mode_sync_phishing(client)


if __name__ == "__main__":
    main()
