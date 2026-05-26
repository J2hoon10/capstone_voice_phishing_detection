"""
OpenAI 배치 작업 완료 후 결과 수집 → phishing_labeled.csv 에 피싱 라벨 추가.

submit_batch.py 가 생성한 batch_state.json 을 읽어:
1. 배치 완료 여부 확인 (--poll 시 자동 대기)
2. 출력 파일 다운로드
3. 각 응답 파싱 + 검증 → phishing_labeled.csv 에 추가 기록
"""
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
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_FIELDS = ["id", "sentence_idx", "sentence", "risk", "label", "category", "source"]


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
        if s["risk"] not in (1, 2, 3):
            return False
        if not isinstance(s["sentence"], str) or not s["sentence"].strip():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI Batch 결과 수집")
    parser.add_argument(
        "--state-file", default=str(OUTPUT_DIR / "batch_state.json"),
        help="submit_batch.py 가 생성한 상태 파일",
    )
    parser.add_argument(
        "--poll", action="store_true",
        help="배치 완료될 때까지 자동 폴링 (최대 24시간 소요 가능)",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="폴링 간격 (초, 기본 60)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다.")

    client = OpenAI(api_key=api_key)

    state_file = Path(args.state_file)
    if not state_file.exists():
        raise SystemExit(f"[ERROR] 상태 파일 없음: {state_file}\n먼저 submit_batch.py 를 실행하세요.")

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)

    batch_id: str = state["batch_id"]
    output_csv = Path(state["output_csv"])
    row_meta: dict[str, dict] = state["row_meta"]

    print(f"[INFO] batch_id  : {batch_id}")
    print(f"[INFO] 제출 시각 : {state.get('submitted_at', '알 수 없음')}")
    print(f"[INFO] 요청 건수 : {state.get('request_count', '?')}건")

    # 배치 완료 대기
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"[INFO] 상태: {batch.status} — "
            f"완료 {counts.completed}/{counts.total} | 실패 {counts.failed}"
        )
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise SystemExit(f"[ERROR] 배치 종료 상태: {batch.status}")
        if not args.poll:
            print(
                "[INFO] 아직 처리 중입니다. 나중에 다시 실행하거나 --poll 로 자동 대기하세요."
            )
            return
        print(f"[INFO] {args.poll_interval}초 후 재확인...")
        time.sleep(args.poll_interval)

    # 오류 파일 출력 (있을 경우)
    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text
        for line in error_text.strip().split("\n"):
            if line.strip():
                err = json.loads(line)
                cid = err.get("custom_id", "?")
                meta = row_meta.get(cid)
                rid = meta["id"] if meta else cid
                print(f"[ERROR] {rid}: {err.get('error', err)}")

    # 결과 다운로드
    result_text = client.files.content(batch.output_file_id).text

    # 이미 출력 CSV에 기록된 id 수집 (일반 데이터 + 이전 실행분)
    done_ids: set[str] = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        with output_csv.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["id"])

    fail_count = 0
    success_count = 0

    with open(output_csv, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)

        for line in result_text.strip().split("\n"):
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            meta = row_meta.get(custom_id)
            if not meta:
                print(f"[WARN] 매핑 없음: {custom_id}")
                fail_count += 1
                continue

            row_id = meta["id"]
            if row_id in done_ids:
                continue

            if result.get("error"):
                print(f"[ERROR] {row_id}: {result['error']}")
                fail_count += 1
                continue

            try:
                raw = result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                print(f"[ERROR] {row_id} 응답 구조 오류: {e}")
                fail_count += 1
                continue

            parsed = parse_response(raw)
            if not parsed or not validate_sentences(parsed):
                print(f"[WARN] {row_id} 파싱 실패. 응답: {raw[:200]}")
                fail_count += 1
                continue

            risk_counts = {1: 0, 2: 0, 3: 0}
            for idx, s in enumerate(parsed):
                risk = int(s["risk"])
                risk_counts[risk] += 1
                writer.writerow({
                    "id": row_id,
                    "sentence_idx": idx,
                    "sentence": s["sentence"],
                    "risk": risk,
                    "label": meta["label"],
                    "category": meta["category"],
                    "source": meta["source"],
                })

            done_ids.add(row_id)
            success_count += 1
            print(
                f"[OK] {row_id} ({meta['category'] or '피싱'}) — "
                f"{len(parsed)}발화 "
                f"위험1:{risk_counts[1]} 위험2:{risk_counts[2]} 위험3:{risk_counts[3]}"
            )

    print(f"\n[DONE] {output_csv}")
    print(f"       성공: {success_count}건 | 실패: {fail_count}건")

    if fail_count > 0:
        print(f"[INFO] 실패 {fail_count}건은 label_sentences.py --resume 으로 재처리할 수 있습니다.")


if __name__ == "__main__":
    main()
