import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")


def clean_text(text: str) -> str:
    cleaned = _PUNCT_RE.sub("", text)
    return re.sub(r" +", " ", cleaned).strip()

try:
    from transformers import AutoTokenizer
except ImportError:
    print("[ERROR] transformers 패키지가 필요합니다. pip install transformers")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

OUTPUT_FIELDS = ["id", "segment_risks", "label", "category", "source"]

SEP = " "


def map_conversation(sentences: list[dict], tokenizer, window_size: int, stride: int) -> list[int]:
    """
    sentences: [{"sentence_idx": int, "sentence": str, "risk": int}, ...]
    Returns list of segment risk values (one per sliding window segment).
    """
    sentences = sorted(sentences, key=lambda s: s["sentence_idx"])

    # Build full text and record character spans per sentence
    char_spans: list[tuple[int, int, int]] = []  # (start, end, risk)
    full_text = ""
    for s in sentences:
        start = len(full_text)
        full_text += clean_text(s["sentence"])
        end = len(full_text)
        char_spans.append((start, end, s["risk"]))
        full_text += SEP

    if not full_text.strip():
        return []

    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    token_strings: list[str] = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    # Assign risk to each token based on character span overlap
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

    # Apply sliding window with word-boundary alignment
    n = len(token_risks)
    segment_risks: list[int] = []
    pos = 0
    while pos < n:
        # 윈도우 끝 위치: 단어 경계까지 확장
        end = min(pos + window_size, n) - 1
        while end + 1 < n and token_strings[end + 1].startswith("##"):
            end += 1

        segment_risks.append(max(token_risks[pos: end + 1]))

        if end + 1 >= n:
            break

        # stride 이동 후 단어 시작 위치로 정렬
        next_pos = pos + stride
        while next_pos < n and token_strings[next_pos].startswith("##"):
            next_pos += 1
        if next_pos >= n:
            break
        pos = next_pos

    return segment_risks


def extract_original_id(asr_id: str) -> str | None:
    """asr_{original_id}_{seq} 형식에서 original_id 추출."""
    m = re.match(r"^asr_(.+)_\d+$", asr_id)
    return m.group(1) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="문장 위험도 → 세그먼트 위험도 매핑")
    parser.add_argument(
        "--labeled-csv",
        default=str(OUTPUT_DIR / "phishing_labeled.csv"),
        help="label_sentences.py 출력 CSV",
    )
    parser.add_argument(
        "--output-csv",
        default=str(OUTPUT_DIR / "segment_risks.csv"),
        help="세그먼트 위험도 출력 CSV",
    )
    parser.add_argument(
        "--model-name",
        default="monologg/koelectra-base-v3-discriminator",
        help="토크나이저 (학습 모델과 동일해야 함)",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument(
        "--append", action="store_true",
        help="기존 output-csv에 이미 있는 ID는 건너뛰고 신규 항목만 추가 (top-up용)",
    )
    args = parser.parse_args()

    print(f"[INFO] 토크나이저 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    conversations: dict[str, list[dict]] = defaultdict(list)
    conv_meta: dict[str, dict] = {}

    with open(args.labeled_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            conv_id = row["id"]
            conversations[conv_id].append(
                {
                    "sentence_idx": int(row["sentence_idx"]),
                    "sentence": row["sentence"],
                    "risk": int(row["risk"]),
                }
            )
            if conv_id not in conv_meta:
                conv_meta[conv_id] = {
                    "label": row.get("label", ""),
                    "category": row.get("category", ""),
                    "source": row.get("source", ""),
                }

    # --append: 기존 파일의 ID 수집 → 신규 항목만 처리
    output_path = Path(args.output_csv)
    done_ids: set[str] = set()
    if args.append and output_path.exists():
        with output_path.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["id"])
        print(f"[INFO] 기존 segment_risks: {len(done_ids)}건 (skip 대상)")

    new_convs = {k: v for k, v in conversations.items() if k not in done_ids}
    print(f"[INFO] {len(new_convs)}개 대화 처리 중 (전체 {len(conversations)}개)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and output_path.exists() else "w"

    with open(output_path, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()

        for conv_id, sentences in new_convs.items():
            seg_risks = map_conversation(sentences, tokenizer, args.window_size, args.stride)
            meta = conv_meta[conv_id]
            writer.writerow(
                {
                    "id": conv_id,
                    "segment_risks": json.dumps(seg_risks),
                    "label": meta["label"],
                    "category": meta["category"],
                    "source": meta["source"],
                }
            )

    total = len(done_ids) + len(new_convs)
    print(f"[DONE] {output_path}  (신규 {len(new_convs)}건 추가, 전체 {total}건)")


if __name__ == "__main__":
    main()
