import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

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
        full_text += s["sentence"]
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

    # Apply sliding window
    n = len(token_risks)
    segment_risks: list[int] = []
    pos = 0
    while pos < n:
        window = token_risks[pos: pos + window_size]
        segment_risks.append(max(window))
        if pos + window_size >= n:
            break
        pos += stride

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
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=100)
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

    print(f"[INFO] {len(conversations)}개 대화 처리 중...")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for conv_id, sentences in conversations.items():
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

    print(f"[DONE] {args.output_csv}  ({len(conversations)}개 대화)")


if __name__ == "__main__":
    main()
