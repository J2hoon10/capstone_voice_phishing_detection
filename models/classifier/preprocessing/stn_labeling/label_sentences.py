import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path


try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai 패키지가 필요합니다. pip install openai")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
OUTPUT_DIR = SCRIPT_DIR / "output"

OUTPUT_FIELDS = ["id", "sentence_idx", "sentence", "risk", "label", "category", "source"]


def load_system_prompt() -> str:
    path = PROMPTS_DIR / "system_prompt.txt"
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def call_api(client: OpenAI, model: str, system_prompt: str, text: str, max_retries: int = 3) -> str:
    prompt = f"다음 보이스피싱 STT 텍스트를 발화 단위로 분리하고 각 단위에 위험도를 부여하라.\n\n[STT 텍스트]\n{text}"
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=prompt,
                max_output_tokens=16384,
            )
            return response.output_text
        except Exception as exc:
            wait_sec = 2 ** attempt * 5
            print(f"[WARN] API 호출 실패: {exc}. {wait_sec}초 후 재시도 ({attempt+1}/{max_retries})")
            time.sleep(wait_sec)
    raise RuntimeError("API 최대 재시도 횟수 초과")


def parse_response(raw: str) -> list[dict] | None:
    raw = raw.strip()
    # 직접 파싱 시도
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # 마크다운 코드블록 내 JSON 추출
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 순수 JSON 배열 추출
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
        if s["risk"] not in (1, 2, 3):
            return False
        if not isinstance(s["sentence"], str) or not s["sentence"].strip():
            return False
    return True


def make_normal_sentences(text: str) -> list[dict]:
    """일반 대화 데이터: 전체를 위험도 1로 단일 처리."""
    return [{"sentence": text, "risk": 1}]


def load_done_ids(output_csv: Path) -> set[str]:
    """이미 처리된 conversation id 수집 (sentence_idx=0 행 기준)."""
    done: set[str] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return done
    with output_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            done.add(row["id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 기반 STT 문장 분리 + 위험도 라벨링")
    parser.add_argument("--input-csv", required=True, nargs="+", help="입력 CSV 파일 경로 (여러 개 가능, id/text/label/category/source 컬럼 필수)")
    parser.add_argument("--output-csv", default=str(OUTPUT_DIR / "phishing_labeled.csv"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--resume", action="store_true", help="기존 출력 파일에서 이어서 실행")
    parser.add_argument("--batch-size", type=int, default=0, help="한 번에 처리할 최대 건수 (0=무제한, --resume과 함께 사용)")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    client = OpenAI(api_key=api_key)
    system_prompt = load_system_prompt()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_csv) if args.resume else set()
    has_file = output_csv.exists() and output_csv.stat().st_size > 0
    mode = "a" if args.resume and has_file else "w"

    seen_ids: set[str] = set()
    rows: list[dict] = []
    for csv_path in args.input_csv:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rid = (row.get("id") or "").strip()
                if rid and rid not in seen_ids:
                    rows.append(row)
                    seen_ids.add(rid)

    phishing_count = sum(1 for r in rows if str(r.get("label", "0")).strip() == "1")
    print(f"[INFO] 전체 {len(rows)}건 — 피싱 {phishing_count}건 / 일반 {len(rows) - phishing_count}건")
    if args.resume:
        print(f"[INFO] 이미 완료: {len(done_ids)}건 → 나머지 {len(rows) - len(done_ids)}건 처리")

    fail_count = 0
    processed = 0
    with open(output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()

        for row in rows:
            if args.batch_size and processed >= args.batch_size:
                remaining = sum(1 for r in rows if (r.get("id") or "").strip() not in done_ids)
                print(f"[INFO] 배치 {args.batch_size}건 완료. 남은 건수: {remaining}건. --resume --batch-size {args.batch_size} 로 이어서 실행하세요.")
                break

            row_id = (row.get("id") or "").strip()
            if row_id in done_ids:
                continue

            text = (row.get("text") or "").strip()
            label = str(row.get("label", "0")).strip()
            category = (row.get("category") or "").strip()
            source = (row.get("source") or "").strip()

            if not text:
                continue

            if label != "1":
                # 일반 데이터: LLM 호출 없이 위험도 1 자동 할당
                sentences = make_normal_sentences(text)
            else:
                # 피싱 데이터: LLM 라벨링
                sentences = None
                for attempt in range(args.max_retries):
                    raw = call_api(client, args.model, system_prompt, text)
                    parsed = parse_response(raw)
                    if parsed and validate_sentences(parsed):
                        sentences = parsed
                        break
                    print(f"[WARN] {row_id} 파싱 실패 (시도 {attempt + 1}/{args.max_retries}). 응답: {raw[:200]}")

                if sentences is None:
                    print(f"[ERROR] {row_id} 라벨링 실패 — 건너뜀")
                    fail_count += 1
                    continue

            risk_counts = {1: 0, 2: 0, 3: 0}
            for idx, s in enumerate(sentences):
                risk = int(s["risk"])
                risk_counts[risk] += 1
                writer.writerow({
                    "id": row_id,
                    "sentence_idx": idx,
                    "sentence": s["sentence"],
                    "risk": risk,
                    "label": label,
                    "category": category,
                    "source": source,
                })
            f.flush()

            done_ids.add(row_id)
            processed += 1
            print(
                f"[OK] {row_id} ({category or '일반'}) — "
                f"{len(sentences)}발화 "
                f"위험도1:{risk_counts[1]} 2:{risk_counts[2]} 3:{risk_counts[3]}"
            )

    print(f"\n[DONE] {output_csv}  (실패: {fail_count}건)")


if __name__ == "__main__":
    main()
