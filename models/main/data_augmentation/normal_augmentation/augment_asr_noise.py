"""
normal_combined.csv 에 ASR 노이즈를 주입해 normal_asr_noised.csv 를 생성합니다.
피싱 증강 파이프라인의 augment_asr_noise.py 와 동일한 노이즈 로직을 사용합니다.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = BASE_DIR.parent
ERROR_SUMMARY_PATH = PREPROCESSING_DIR / "error_analysis" / "error_summary.json"
OUT_DIR = BASE_DIR / "output"

# ── 한글 분해/조합 테이블 ──────────────────────────────────────────────────
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
    s = code - 0xAC00
    return s // 588, (s % 588) // 28, s % 28


def compose(i, m, f):
    return chr(0xAC00 + i * 588 + m * 28 + f)


def substitute_hangul(ch, rng):
    dec = decompose(ch)
    if dec is None:
        return rng.choice(INSERT_TOKENS)
    i, m, f = dec
    candidates = []
    if INITIALS[i] in SIMILAR_INITIAL:
        for ni in SIMILAR_INITIAL[INITIALS[i]]:
            candidates.append((INITIALS.index(ni), m, f))
    if MEDIALS[m] in SIMILAR_MEDIAL:
        for nm in SIMILAR_MEDIAL[MEDIALS[m]]:
            candidates.append((i, MEDIALS.index(nm), f))
    if FINALS[f] in SIMILAR_FINAL:
        for nf in SIMILAR_FINAL[FINALS[f]]:
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
    for ch in text:
        if ch == " ":
            if rng.random() < p_space:
                continue
            out.append(ch)
            continue
        if rng.random() < p_del:
            continue
        out.append(substitute_hangul(ch, rng) if rng.random() < p_sub else ch)
        if rng.random() < p_ins:
            out.append(rng.choice(INSERT_TOKENS))
        if rng.random() < p_space / 2:
            out.append(" ")
    return "".join(out)


SAMPLES_PER_TRACK = 100


def sample_by_track(rows: list, n: int, rng: random.Random) -> list:
    """트랙별로 n개씩 무작위 샘플링."""
    by_track: dict[str, list] = {}
    for row in rows:
        t = (row.get("track") or "").strip().upper()
        by_track.setdefault(t, []).append(row)
    selected = []
    for track, track_rows in sorted(by_track.items()):
        shuffled = list(track_rows)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:n])
        print(f"  [{track}] 원본 {len(track_rows)}개 중 {min(n, len(track_rows))}개 선택")
    return selected


def main():
    parser = argparse.ArgumentParser(description="원본+증강 트랙별 350개 샘플 ASR 노이즈 주입")
    parser.add_argument(
        "--input-original",
        default=str(OUT_DIR / "normal_combined.csv"),
        help="원본 CSV (id, script, label, track) — 트랙별 350개 무작위 샘플링",
    )
    parser.add_argument(
        "--input-augmented",
        default=str(OUT_DIR / "normal_augmented.csv"),
        help="증강 CSV (id, script, label, track) — 전량 사용",
    )
    parser.add_argument(
        "--error-summary",
        default=str(ERROR_SUMMARY_PATH),
        help="오류 요약 JSON",
    )
    parser.add_argument(
        "--output",
        default=str(OUT_DIR / "normal_asr_noised.csv"),
        help="출력 CSV",
    )
    parser.add_argument("--samples-per-track", type=int, default=SAMPLES_PER_TRACK,
                        help="원본에서 트랙별 샘플 수 (기본 350)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="노이즈 확률 배율 (0.5 = 절반, 1.0 = 원본)")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    with open(args.error_summary, "r", encoding="utf-8") as f:
        probs = json.load(f)

    _noise_keys = {"substitution", "deletion", "insertion", "whitespace"}
    probs = {k: (v * args.noise_scale if k in _noise_keys else v) for k, v in probs.items()}

    rng = random.Random(args.seed)

    # 원본: 트랙별 350개 샘플링
    print("[INFO] 원본 데이터 샘플링...")
    original_all = []
    with open(args.input_original, "r", encoding="utf-8-sig") as f:
        original_all = list(csv.DictReader(f))
    original_rows = sample_by_track(original_all, args.samples_per_track, rng)

    # 증강: 전량 사용
    print("[INFO] 증강 데이터 로드...")
    augmented_rows = []
    with open(args.input_augmented, "r", encoding="utf-8-sig") as f:
        augmented_rows = list(csv.DictReader(f))
    by_track_aug = {}
    for r in augmented_rows:
        t = (r.get("track") or "").strip().upper()
        by_track_aug[t] = by_track_aug.get(t, 0) + 1
    for t, cnt in sorted(by_track_aug.items()):
        print(f"  [{t}] 증강 {cnt}개 전량 사용")

    rows = original_rows + augmented_rows
    if not rows:
        raise SystemExit("[ERROR] 입력 데이터가 없습니다.")

    print(f"[INFO] 총 {len(rows)}개 노이즈 적용 (원본 {len(original_rows)} + 증강 {len(augmented_rows)})")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    out_rows = []
    iterator = rows if (args.no_progress or tqdm is None) else tqdm(rows, desc="asr-noise", unit="row")
    for idx, row in enumerate(iterator, 1):
        script = (row.get("script") or "").strip()
        if not script:
            continue
        noised = apply_noise(script, probs, rng)
        out_rows.append({
            "id": f"asr_{row.get('id', '')}",
            "script": noised,
            "label": row.get("label", "0"),
            "track": row.get("track", ""),
        })

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "script", "label", "track"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] {args.output} ({len(out_rows)}건)")


if __name__ == "__main__":
    main()
