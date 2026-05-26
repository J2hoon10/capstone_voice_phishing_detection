"""
compute_segment_risks.py

normal_combined.csv(노이즈 전)의 각 스크립트를 토크나이징한 뒤
슬라이딩 윈도우를 적용하여 segment_risks 배열을 생성합니다.
일반 대화(label=0)이므로 모든 위험도는 1입니다.

map_segments.py (stn_labeling) 와 동일한 tokenizer / window 파라미터를 사용합니다.
출력 포맷도 segment_risks.csv 와 동일합니다:
  id, segment_risks, label, category, source
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]")


def clean_text(text: str) -> str:
    cleaned = _PUNCT_RE.sub("", text)
    return re.sub(r" +", " ", cleaned).strip()

try:
    from transformers import AutoTokenizer
except ImportError:
    print("[ERROR] transformers 패키지가 필요합니다: pip install transformers")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "output"


def compute_segment_risks_all_one(text: str, tokenizer, window_size: int, stride: int) -> list[int]:
    """텍스트를 토크나이징하고 슬라이딩 윈도우 수만큼 1로 채운 배열을 반환합니다."""
    if not text.strip():
        return []

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    n = len(encoding["offset_mapping"])
    if n == 0:
        return []

    token_strings: list[str] = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    segment_risks: list[int] = []
    pos = 0
    while pos < n:
        # 윈도우 끝 위치: 단어 경계까지 확장
        end = min(pos + window_size, n) - 1
        while end + 1 < n and token_strings[end + 1].startswith("##"):
            end += 1

        segment_risks.append(1)  # 일반 대화: 항상 위험도 1

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


def main():
    parser = argparse.ArgumentParser(description="일반 대화 segment_risks 생성 (모두 1)")
    parser.add_argument(
        "--input",
        default=str(OUT_DIR / "normal_combined.csv"),
        help="입력 CSV (노이즈 전, script 컬럼 사용)",
    )
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "normal_segment_risks.csv"),
        help="출력 CSV",
    )
    parser.add_argument(
        "--model-name",
        default="monologg/koelectra-base-v3-discriminator",
        help="토크나이저 (map_segments.py 와 동일하게 맞출 것)",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride",      type=int, default=32)
    args = parser.parse_args()

    print(f"[INFO] 토크나이저 로드: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    rows_in = []
    with open(args.input, encoding="utf-8-sig") as f:
        rows_in = list(csv.DictReader(f))

    if not rows_in:
        raise SystemExit(f"[ERROR] 입력 파일이 비어 있습니다: {args.input}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "segment_risks", "label", "category", "source"]
        )
        writer.writeheader()

        for row in rows_in:
            text = clean_text((row.get("script") or "").strip())
            seg_risks = compute_segment_risks_all_one(text, tokenizer, args.window_size, args.stride)
            writer.writerow({
                "id":            row["id"],
                "segment_risks": json.dumps(seg_risks),
                "label":         row.get("label", "0"),
                "category":      row.get("track", ""),   # track(A/B)을 category 위치에 저장
                "source":        "normal_combined",
            })
            written += 1

    print(f"[DONE] {args.output}  ({written}개 대화)")
    # 샘플 확인
    with open(args.output, encoding="utf-8-sig") as f:
        sample = list(csv.DictReader(f))[:2]
    for s in sample:
        risks = json.loads(s["segment_risks"])
        print(f"  {s['id']}  segments={len(risks)}  값 예시={risks[:5]}")


if __name__ == "__main__":
    main()
