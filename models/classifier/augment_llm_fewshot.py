# augment_llm_fewshot.py
# 피싱 seed 문장에서 LLM으로 구어체 변형 생성

import os
import csv
import json
import time
import argparse
import random
import re

from pipeline_config import (
    TRANSCRIPTION_DIR,
    AUGMENTED_DIR,
    DEFAULT_VARIANT,
    LLM_AUGMENT_RATIO,
    MIN_TEXT_LENGTH,
    MIN_KOREAN_RATIO,
    CSV_COLUMNS,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def load_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def korean_ratio(text):
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    kor = sum(1 for c in chars if "가" <= c <= "힣")
    return kor / len(chars)


def is_valid(text):
    t = (text or "").strip()
    if len(t) < MIN_TEXT_LENGTH:
        return False
    if korean_ratio(t) < MIN_KOREAN_RATIO:
        return False
    return True


def extract_json_array(text):
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return []
    return []


def call_openai_compat(prompt, api_base, api_key, model, temperature=0.7, max_retries=3):
    import requests

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "너는 한국어 피싱 문장을 구어체로 변형하는 보조자다."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        time.sleep(1 + attempt)

    raise RuntimeError(f"LLM 호출 실패: {resp.status_code} {resp.text[:200]}")


def mock_variations(text, n, rng):
    variants = []
    fillers = ["저기", "그", "지금", "혹시", "아", "음", "그러니까", "잠깐만"]
    endings = [("입니다", "이에요"), ("하세요", "해요"), ("드립니다", "드려요"), ("하십시오", "하세요")]

    for _ in range(n):
        t = text
        if rng.random() < 0.6:
            t = rng.choice(fillers) + ", " + t
        for a, b in endings:
            if a in t and rng.random() < 0.7:
                t = t.replace(a, b)
        if rng.random() < 0.5 and not t.endswith("요"):
            t = t + "요"
        variants.append(t)
    return variants


def build_prompt(text, n):
    return (
        "다음 문장의 의미와 의도는 유지하고, 한국어 구어체로 표현만 바꾼 문장을 "
        f"{n}개 만들어줘.\n"
        "출력 형식은 반드시 JSON 배열 (문장 문자열만)로 해줘.\n"
        f"문장: {text}"
    )


def main():
    parser = argparse.ArgumentParser(description="LLM 구어체 증강 생성")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Whisper 변형")
    parser.add_argument("--input", default=None, help="입력 CSV 경로 (기본: phishing.csv)")
    parser.add_argument("--output", default=None, help="출력 CSV 경로")
    parser.add_argument("--ratio", type=int, default=LLM_AUGMENT_RATIO, help="원본 1개당 생성 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--api-base", default=os.getenv("LLM_API_BASE"), help="OpenAI-호환 API Base")
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY"), help="API Key")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"), help="모델 이름")
    parser.add_argument("--provider", default=None, choices=["openai_compat", "mock"], help="LLM 제공자")
    parser.add_argument("--no-progress", action="store_true", help="진행률 표시 비활성화")
    args = parser.parse_args()

    input_csv = args.input or os.path.join(TRANSCRIPTION_DIR, args.variant, "phishing.csv")
    output_csv = args.output or os.path.join(AUGMENTED_DIR, "llm_fewshot.csv")

    rows = load_rows(input_csv)
    if not rows:
        raise SystemExit(f"입력 CSV를 찾을 수 없거나 비어있음: {input_csv}")

    rng = random.Random(args.seed)

    # 시드 문장 품질 필터링 + 중복 제거
    seeds = []
    seen = set()
    for row in rows:
        text = (row.get("text") or "").strip()
        if not is_valid(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        seeds.append(row)

    if not seeds:
        raise SystemExit("유효한 seed 문장을 찾지 못했습니다.")

    provider = args.provider
    if provider is None:
        provider = "openai_compat" if args.api_key else "mock"

    if provider == "openai_compat":
        if not (args.api_base and args.api_key and args.model):
            raise SystemExit("OpenAI-호환 API 설정이 필요합니다. --api-base/--api-key/--model 또는 환경변수 설정")

    os.makedirs(AUGMENTED_DIR, exist_ok=True)

    out_rows = []

    iterator = seeds
    if tqdm and not args.no_progress:
        iterator = tqdm(seeds, total=len(seeds), desc="llm-augment", unit="seed")

    for idx, row in enumerate(iterator, 1):
        text = row["text"].strip()
        prompt = build_prompt(text, args.ratio)

        if provider == "openai_compat":
            response = call_openai_compat(prompt, args.api_base, args.api_key, args.model)
            variations = extract_json_array(response)
        else:
            variations = mock_variations(text, args.ratio, rng)

        clean_vars = []
        for v in variations:
            v = (v or "").strip()
            if not is_valid(v):
                continue
            clean_vars.append(v)

        # 중복 제거
        clean_vars = list(dict.fromkeys(clean_vars))

        for j, v in enumerate(clean_vars, 1):
            out_rows.append({
                "id": f"llm_{row.get('category','unknown')}_{row.get('id','')}_{j}",
                "text": v,
                "label": 1,
                "category": row.get("category", ""),
                "source": "llm_fewshot",
                "filename": row.get("filename", ""),
            })

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"LLM 증강 저장: {output_csv} ({len(out_rows)}건)")


if __name__ == "__main__":
    main()
