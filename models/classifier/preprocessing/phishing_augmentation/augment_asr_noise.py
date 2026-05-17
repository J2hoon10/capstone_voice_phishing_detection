# augment_asr_noise.py
# 오류 확률 기반 ASR 노이즈 주입

import os
import csv
import json
import argparse
import random

from pipeline_config import (
    AUGMENTED_DIR,
    ERROR_ANALYSIS_DIR,
    DEFAULT_VARIANT,
    CSV_COLUMNS,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

INITIALS = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
MEDIALS = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ",
    "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
FINALS = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
    "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

SIMILAR_INITIAL = {
    "ㄱ": ["ㅋ"], "ㅋ": ["ㄱ"],
    "ㄷ": ["ㅌ"], "ㅌ": ["ㄷ"],
    "ㅂ": ["ㅍ"], "ㅍ": ["ㅂ"],
    "ㅈ": ["ㅊ"], "ㅊ": ["ㅈ"],
    "ㅅ": ["ㅆ"], "ㅆ": ["ㅅ"],
}
SIMILAR_MEDIAL = {
    "ㅐ": ["ㅔ"], "ㅔ": ["ㅐ"],
    "ㅏ": ["ㅓ"], "ㅓ": ["ㅏ"],
    "ㅗ": ["ㅜ"], "ㅜ": ["ㅗ"],
    "ㅡ": ["ㅜ", "ㅗ"],
    "ㅣ": ["ㅟ", "ㅢ"],
}
SIMILAR_FINAL = {
    "ㄱ": ["ㅋ"], "ㅋ": ["ㄱ"],
    "ㄷ": ["ㅌ"], "ㅌ": ["ㄷ"],
    "ㅂ": ["ㅍ"], "ㅍ": ["ㅂ"],
    "ㅅ": ["ㅆ"], "ㅆ": ["ㅅ"],
}

INSERT_TOKENS = ["음", "어", "아", "그", "저", "네", "요"]


def decompose(ch):
    code = ord(ch)
    if code < 0xAC00 or code > 0xD7A3:
        return None
    s_index = code - 0xAC00
    i = s_index // 588
    m = (s_index % 588) // 28
    f = s_index % 28
    return i, m, f


def compose(i, m, f):
    return chr(0xAC00 + (i * 588) + (m * 28) + f)


def substitute_hangul(ch, rng):
    dec = decompose(ch)
    if dec is None:
        return rng.choice(INSERT_TOKENS)

    i, m, f = dec
    candidates = []

    init = INITIALS[i]
    med = MEDIALS[m]
    fin = FINALS[f]

    if init in SIMILAR_INITIAL:
        for ni in SIMILAR_INITIAL[init]:
            candidates.append((INITIALS.index(ni), m, f))
    if med in SIMILAR_MEDIAL:
        for nm in SIMILAR_MEDIAL[med]:
            candidates.append((i, MEDIALS.index(nm), f))
    if fin in SIMILAR_FINAL:
        for nf in SIMILAR_FINAL[fin]:
            candidates.append((i, m, FINALS.index(nf)))

    if not candidates:
        return ch

    ni, nm, nf = rng.choice(candidates)
    return compose(ni, nm, nf)


def apply_noise(text, probs, rng):
    if not text:
        return text

    p_sub = min(max(probs.get("substitution", 0.0), 0.0), 0.5)
    p_del = min(max(probs.get("deletion", 0.0), 0.0), 0.5)
    p_ins = min(max(probs.get("insertion", 0.0), 0.0), 0.5)
    p_space = min(max(probs.get("whitespace", 0.0), 0.0), 0.5)

    out = []
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == " ":
            if rng.random() < p_space:
                i += 1
                continue
            out.append(ch)
            i += 1
            continue

        if rng.random() < p_del:
            i += 1
            continue

        if rng.random() < p_sub:
            out.append(substitute_hangul(ch, rng))
        else:
            out.append(ch)

        if rng.random() < p_ins:
            out.append(rng.choice(INSERT_TOKENS))

        if rng.random() < (p_space / 2):
            out.append(" ")

        i += 1

    return "".join(out)


def load_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="ASR 노이즈 주입")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Whisper 변형")
    parser.add_argument("--error-summary", default=os.path.join(ERROR_ANALYSIS_DIR, "error_summary.json"),
                        help="오류 요약 JSON")
    parser.add_argument("--input-llm", default=os.path.join(AUGMENTED_DIR, "phishing_augmented.csv"), help="LLM 증강 CSV")
    parser.add_argument("--input-original", default="", help="원본 전사 CSV (생략 시 증강 데이터만 처리)")
    parser.add_argument("--output", default=os.path.join(AUGMENTED_DIR, "asr_noised.csv"), help="출력 CSV")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="노이즈 확률 배율 (0.5 = 절반, 1.0 = 원본)")
    parser.add_argument("--no-progress", action="store_true", help="진행률 표시 비활성화")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="출력 파일에 이미 존재하는 기반 ID(asr_ 접두사 제거)는 건너뜀 (top-up용)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="출력 파일에 덧붙이기 (skip-existing과 함께 사용하면 기존 파일 유지 + 신규 행 추가)",
    )
    args = parser.parse_args()

    with open(args.error_summary, "r", encoding="utf-8") as f:
        probs = json.load(f)

    _noise_keys = {"substitution", "deletion", "insertion", "whitespace"}
    probs = {k: (v * args.noise_scale if k in _noise_keys else v) for k, v in probs.items()}

    original_rows = load_rows(args.input_original) if args.input_original else []
    llm_rows = load_rows(args.input_llm)

    combined = original_rows + llm_rows
    if not combined:
        raise SystemExit("입력 데이터가 없습니다. --input-llm 또는 --input-original 을 확인하세요.")

    # --skip-existing: 이미 처리된 기반 ID 수집 후 필터링
    if args.skip_existing and os.path.exists(args.output):
        existing_base_ids: set[str] = set()
        with open(args.output, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw_id = row.get("id", "")
                # 형식: asr_<base_id>_<idx>  → 마지막 _숫자 제거 후 asr_ 제거
                base = raw_id
                if base.startswith("asr_"):
                    base = base[4:]
                base = base.rsplit("_", 1)[0] if base.rsplit("_", 1)[-1].isdigit() else base
                existing_base_ids.add(base)
        before = len(combined)
        combined = [r for r in combined if r.get("id", "") not in existing_base_ids]
        print(f"[skip-existing] {before}건 중 {before - len(combined)}건 건너뜀 → {len(combined)}건 처리")

    if not combined:
        print("[INFO] 신규 처리 대상 없음.")
        return

    rng = random.Random(args.seed)

    os.makedirs(AUGMENTED_DIR, exist_ok=True)

    out_rows = []

    iterator = combined
    if tqdm and not args.no_progress:
        iterator = tqdm(combined, total=len(combined), desc="asr-noise", unit="row")

    for idx, row in enumerate(iterator, 1):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        noised = apply_noise(text, probs, rng)
        out_rows.append({
            "id": f"asr_{row.get('id','')}_{idx}",
            "text": noised,
            "label": row.get("label", ""),
            "category": row.get("category", ""),
            "source": "asr_noise",
            "filename": row.get("filename", ""),
        })

    if args.append and os.path.exists(args.output):
        existing_out = load_rows(args.output)
        start_idx = len(existing_out) + 1
        # idx를 기존 건수 이후부터 부여
        for i, row in enumerate(out_rows):
            parts = row["id"].rsplit("_", 1)
            row["id"] = f"{parts[0]}_{start_idx + i}"
        all_out = existing_out + out_rows
        with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_out)
        print(f"ASR 노이즈 추가 저장: {args.output} (+{len(out_rows)}건, 전체 {len(all_out)}건)")
    else:
        with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"ASR 노이즈 저장: {args.output} ({len(out_rows)}건)")


if __name__ == "__main__":
    main()
