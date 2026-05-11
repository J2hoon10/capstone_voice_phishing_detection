import argparse
import csv
import os
import random
import re
import sys
import time
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai 패키지가 필요합니다. pip install openai")
    sys.exit(1)

from pipeline_config import LEGACY_AUGMENTED_DIR, TRANSCRIPTION_DIR, DEFAULT_VARIANT
from prompt_loader import load_category_prompts, load_system_prompt

OUTPUT_FIELDS = ["id", "text", "label", "category", "source"]


def build_prompt(category: str, few_shots: list[str], n_generate: int, category_prompt: str) -> str:
    examples = "\n\n".join(
        f"[실제 STT 예시 {i+1}]\n{sample}" for i, sample in enumerate(few_shots)
    )
    return (
        f"{category_prompt}\n\n"
        f"[실제 STT 예시]\n{examples}\n\n"
        f"위 예시 패턴을 참고해 동일 카테고리 보이스피싱 STT 전사 텍스트를 {n_generate}개 생성하라.\n"
        "각 결과는 반드시 [변형1], [변형2] ... 태그로 구분하라."
    )


def parse_variants(response_text: str) -> list[str]:
    parts = re.split(r"\[변형\d+\]", response_text)
    return [p.strip() for p in parts if p.strip()]


def looks_victim_initiated(text: str) -> bool:
    head = text[:140].replace(" ", "")
    victim_like_starts = [
        "제가대출문의",
        "대출문의드리려고",
        "제가신청",
        "문의드리려고전화",
        "상담받으려고전화",
        "제가먼저전화",
    ]
    return any(token in head for token in victim_like_starts)


def call_api(client: OpenAI, model: str, system_prompt: str, prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=prompt,
                max_output_tokens=4096,
            )
            return response.output_text
        except Exception as exc:
            wait_sec = 2 ** attempt * 5
            print(f"[WARN] API 호출 실패: {exc}. {wait_sec}초 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait_sec)
    raise RuntimeError("API 최대 재시도 횟수 초과")


def load_done_counts(csv_path: str) -> dict[str, int]:
    counts = defaultdict(int)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return counts
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            counts[row["category"]] += 1
    return counts


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


def main():
    parser = argparse.ArgumentParser(description="OpenAI API 기반 피싱 텍스트 증강")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--n-per-category", type=int, default=200)
    parser.add_argument(
        "--target-total-per-category",
        type=int,
        default=500,
        help="카테고리별 최종 목표 건수",
    )
    parser.add_argument("--per-call", type=int, default=5)
    parser.add_argument("--few-shot-k", type=int, default=10)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-csv", default=os.path.join(LEGACY_AUGMENTED_DIR, "phishing_augmented.csv"))
    parser.add_argument("--fewshot-csv", default="", help="정제된 few-shot CSV 경로 (컬럼: category,text_clean)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    random.seed(args.seed)
    client = OpenAI(api_key=api_key)
    system_prompt = load_system_prompt()
    category_prompts = load_category_prompts()

    source_texts: dict[str, list[str]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)
    source_csv = os.path.join(TRANSCRIPTION_DIR, args.variant, "phishing.csv")
    with open(source_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            category = (row.get("category") or "").strip()
            text = (row.get("text") or "").strip()
            if category:
                source_counts[category] += 1
            if category and 1000 <= len(text) <= 2000:
                source_texts[category].append(text)

    if args.fewshot_csv:
        source_texts = load_fewshot_from_clean_csv(args.fewshot_csv)
        print(f"[INFO] 정제 few-shot 사용: {args.fewshot_csv}")
    else:
        print(f"[INFO] 원본 STT few-shot 사용: {source_csv}")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    done_counts = load_done_counts(args.output_csv) if args.resume else {}
    has_file = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    mode = "a" if args.resume and has_file else "w"

    with open(args.output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()

        for category, category_prompt in category_prompts.items():
            target_generate = max(args.target_total_per_category - source_counts.get(category, 0), 0)

            already_done = done_counts.get(category, 0)
            remaining = target_generate - already_done
            if remaining <= 0:
                print(f"[SKIP] {category}: 목표 충족 (원본 {source_counts.get(category, 0)}건, 증강목표 {target_generate}건)")
                continue

            candidates = source_texts.get(category, [])
            if not candidates:
                print(f"[SKIP] few-shot 후보 없음: {category}")
                continue

            few_shots = random.sample(candidates, min(args.few_shot_k, len(candidates)))
            generated: list[str] = []
            while len(generated) < remaining:
                batch_n = min(args.per_call, remaining - len(generated))
                prompt = build_prompt(category, few_shots, batch_n, category_prompt)
                raw = call_api(client, args.model, system_prompt, prompt)
                variants = parse_variants(raw)
                if not variants:
                    print(f"[WARN] 파싱 실패. 응답 일부: {raw[:200]}")
                    continue
                filtered = [v for v in variants if not looks_victim_initiated(v)]
                if not filtered:
                    print("[WARN] 발화 주체 규칙 위반(피해자 시작)으로 재시도")
                    continue
                generated.extend(filtered[:batch_n])

            for idx, text in enumerate(generated, start=already_done + 1):
                writer.writerow(
                    {
                        "id": f"aug_{category}_{idx:04d}",
                        "text": text,
                        "label": 1,
                        "category": category,
                        "source": "augmented_llm",
                    }
                )
            f.flush()
            print(f"[OK] {category}: {len(generated)}건 생성")

    print(f"[DONE] {args.output_csv}")


if __name__ == "__main__":
    main()
