"""
OpenAI Batch API를 사용한 피싱 STT 문장 분리 + 위험도 라벨링 제출.

- label=0 (일반): 로컬 처리 → risk=1 일괄 할당 후 즉시 출력 CSV에 기록
- label=1 (피싱): Batch API 제출 → batch_state.json 에 batch_id 저장

결과 수집은 collect_batch.py 로 수행.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
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
PROMPT_PREFIX = (
    "다음 보이스피싱 STT 텍스트를 발화 단위로 분리하고 각 단위에 위험도를 부여하라.\n\n[STT 텍스트]\n"
)


def load_system_prompt() -> str:
    path = PROMPTS_DIR / "system_prompt.txt"
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def load_done_ids(output_csv: Path) -> set[str]:
    done: set[str] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return done
    with output_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            done.add(row["id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI Batch API 피싱 라벨링 제출")
    parser.add_argument(
        "--input-csv", required=True, nargs="+",
        help="입력 CSV (id/text/label/category/source 컬럼 필수, 여러 개 가능)",
    )
    parser.add_argument(
        "--output-csv", default=str(OUTPUT_DIR / "phishing_labeled.csv"),
        help="일반 데이터 및 최종 피싱 결과가 기록될 CSV",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--resume", action="store_true",
        help="기존 출력 CSV를 읽어 이미 처리된 ID 스킵 (이어서 제출)",
    )
    parser.add_argument(
        "--state-file", default=str(OUTPUT_DIR / "batch_state.json"),
        help="batch_id 및 요청 메타 저장 파일",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    client = OpenAI(api_key=api_key)
    system_prompt = load_system_prompt()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    state_file = Path(args.state_file)

    done_ids = load_done_ids(output_csv) if args.resume else set()

    # 입력 CSV 로드 (id 기준 중복 제거)
    seen_ids: set[str] = set()
    rows: list[dict] = []
    for csv_path in args.input_csv:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rid = (row.get("id") or "").strip()
                if rid and rid not in seen_ids:
                    rows.append(row)
                    seen_ids.add(rid)

    total_phishing = sum(1 for r in rows if str(r.get("label", "0")).strip() == "1")
    total_normal = len(rows) - total_phishing
    print(f"[INFO] 전체 {len(rows)}건 — 피싱 {total_phishing}건 / 일반 {total_normal}건")
    print(f"[INFO] 이미 처리 완료: {len(done_ids)}건")

    normal_todo = [
        r for r in rows
        if str(r.get("label", "0")).strip() != "1"
        and (r.get("id") or "").strip() not in done_ids
    ]
    phishing_todo = [
        r for r in rows
        if str(r.get("label", "0")).strip() == "1"
        and (r.get("id") or "").strip() not in done_ids
    ]
    print(f"[INFO] 처리 대상 — 피싱 배치 {len(phishing_todo)}건 / 일반 로컬 {len(normal_todo)}건")

    # 일반 데이터: 로컬 처리 (risk=1 일괄)
    has_file = output_csv.exists() and output_csv.stat().st_size > 0
    mode = "a" if args.resume and has_file else "w"
    with open(output_csv, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()
        for row in normal_todo:
            row_id = (row.get("id") or "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            writer.writerow({
                "id": row_id,
                "sentence_idx": 0,
                "sentence": text,
                "risk": 1,
                "label": str(row.get("label", "0")).strip(),
                "category": (row.get("category") or "").strip(),
                "source": (row.get("source") or "").strip(),
            })
    print(f"[INFO] 일반 데이터 {len(normal_todo)}건 로컬 처리 완료 → {output_csv}")

    if not phishing_todo:
        print("[INFO] 피싱 배치 제출 대상 없음. 종료.")
        return

    # 배치 요청 JSONL 생성 (custom_id = req_NNNNNN, 인덱스 기반)
    batch_requests_path = output_csv.parent / "batch_requests.jsonl"
    row_meta: dict[str, dict] = {}

    with open(batch_requests_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(phishing_todo):
            row_id = (row.get("id") or "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            custom_id = f"req_{idx:06d}"
            row_meta[custom_id] = {
                "id": row_id,
                "label": str(row.get("label", "1")).strip(),
                "category": (row.get("category") or "").strip(),
                "source": (row.get("source") or "").strip(),
            }
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": PROMPT_PREFIX + text},
                    ],
                    "max_tokens": 16384,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(f"[INFO] 배치 요청 파일 생성: {batch_requests_path} ({len(row_meta)}건)")

    # 파일 업로드
    with open(batch_requests_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"[INFO] 파일 업로드 완료: {uploaded.id}")

    # 배치 제출
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"\n[OK] 배치 제출 완료!")
    print(f"     batch_id  : {batch.id}")
    print(f"     상태      : {batch.status}")
    print(f"     요청 건수 : {len(row_meta)}건")
    print(f"     예상 비용 : gpt-4o-mini 기준 실시간 대비 50% 절감")

    # 상태 파일 저장
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "model": args.model,
        "submitted_at": datetime.now().isoformat(),
        "request_count": len(row_meta),
        "output_csv": str(output_csv),
        "row_meta": row_meta,
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 상태 파일 저장: {state_file}")
    print(f"[INFO] 결과 수집 명령 (완료 후 실행):")
    print(f"  python {SCRIPT_DIR / 'collect_batch.py'} --state-file \"{state_file}\"")
    print(f"  # 또는 완료될 때까지 자동 대기:")
    print(f"  python {SCRIPT_DIR / 'collect_batch.py'} --state-file \"{state_file}\" --poll")


if __name__ == "__main__":
    main()
