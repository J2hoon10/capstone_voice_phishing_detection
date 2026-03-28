"""
MELD 데이터 전처리 모듈

1. HuggingFace Datasets로 MELD 데이터 로드 (실패 시 공식 CSV fallback)
2. 씬(Dialogue) 단위 텍스트 합성
3. 10차원 multi-hot 레이블 생성 (감정 7 + 감성 3)
4. 토큰 기반 세그먼테이션
5. 세그먼트 통계 분석 (r, k 설정 검증용)
"""

import json
from pathlib import Path
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from config import (
    DATA_DIR, MELD_CONFIG, ENCODER_CONFIG
)


# ── 레이블 매핑 테이블 ─────────────────────────────────────
# MELD 원본 레이블 문자열 → multi-hot 인덱스
EMOTION_TO_IDX = {label: i for i, label in enumerate(MELD_CONFIG["EMOTION_LABELS"])}
SENTIMENT_TO_IDX = {label: i + 7 for i, label in enumerate(MELD_CONFIG["SENTIMENT_LABELS"])}

# HuggingFace split명 매핑 (내부 split명 → HuggingFace split명)
HF_SPLIT_MAP = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}

CSV_SPLIT_MAP = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

MELD_CSV_BASE_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"


def _validate_and_normalize_meld_df(df: pd.DataFrame) -> pd.DataFrame:
    """MELD annotation DataFrame의 필수 컬럼과 기본 타입을 정리한다."""
    df = df.copy()
    df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]

    required = ["Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion", "Sentiment"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    for col in ["Speaker", "Utterance", "Emotion", "Sentiment"]:
        df[col] = df[col].fillna("").astype(str)

    for col in ["Dialogue_ID", "Utterance_ID"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    return df


def _load_meld_csv_fallback(split: str) -> pd.DataFrame:
    """
    공식 MELD GitHub 저장소의 CSV annotation 파일로 fallback 로드한다.

    HuggingFace `declare-lab/MELD` 로더가 `webdataset`/tar streaming 이슈로
    실패하는 환경에서 안정적으로 텍스트 annotation만 확보하기 위한 우회 경로다.
    """
    csv_name = CSV_SPLIT_MAP[split]
    cache_dir = DATA_DIR / "_meld_csv_cache"
    csv_path = cache_dir / csv_name

    if not csv_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = f"{MELD_CSV_BASE_URL}/{csv_name}"
        print(f"[Fallback] 공식 CSV 다운로드 중: {url}")
        try:
            urlretrieve(url, csv_path)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                f"MELD fallback CSV 다운로드 실패: {url}. "
                "네트워크 연결 또는 원격 파일 경로를 확인하세요."
            ) from exc
    else:
        print(f"[Fallback] 캐시된 CSV 사용: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return _validate_and_normalize_meld_df(df)


def load_meld_hf(split: str) -> pd.DataFrame:
    """
    HuggingFace Datasets에서 MELD 데이터를 로드한다.
    CSV 다운로드 없이 자동으로 데이터를 가져온다.

    Args:
        split: "train", "dev", "test" 중 하나

    Returns:
        DataFrame (Dialogue_ID, Utterance_ID, Speaker, Utterance, Emotion, Sentiment 포함)
    """
    hf_split = HF_SPLIT_MAP[split]
    print(f"[HuggingFace] declare-lab/MELD ({hf_split}) 로드 중...")

    try:
        dataset = load_dataset("declare-lab/MELD", split=hf_split)
        df = dataset.to_pandas()
        df = _validate_and_normalize_meld_df(df)
        print(f"[HuggingFace] {split} 로드 완료: {len(df)}개 발화")
        return df
    except Exception as exc:
        print(f"[HuggingFace] 로드 실패, 공식 CSV fallback 사용: {exc}")
        df = _load_meld_csv_fallback(split)

    print(f"[Fallback] {split} 로드 완료: {len(df)}개 발화")
    return df


def build_scene_data(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame을 씬(Dialogue) 단위로 그룹핑하여
    텍스트 스크립트 + multi-hot 레이블을 생성한다.

    Args:
        df: MELD DataFrame

    Returns:
        list of dict: [{"dialogue_id": int, "text": str, "label": list[int]}, ...]
    """
    scenes = []
    fmt = MELD_CONFIG["UTTERANCE_FORMAT"]
    sep = MELD_CONFIG["UTTERANCE_SEP"]

    for dialogue_id, group in df.groupby("Dialogue_ID"):
        # 발화 순서대로 정렬
        group = group.sort_values("Utterance_ID")

        # "Speaker: Utterance" 형식으로 이어붙이기
        utterances = []
        for _, row in group.iterrows():
            text = fmt.format(speaker=row["Speaker"], utterance=row["Utterance"])
            utterances.append(text)
        script = sep.join(utterances)

        # 10차원 multi-hot 벡터 생성
        label = [0] * MELD_CONFIG["NUM_LABELS"]
        for _, row in group.iterrows():
            emotion = row["Emotion"].lower().strip()
            sentiment = row["Sentiment"].lower().strip()

            if emotion in EMOTION_TO_IDX:
                label[EMOTION_TO_IDX[emotion]] = 1
            if sentiment in SENTIMENT_TO_IDX:
                label[SENTIMENT_TO_IDX[sentiment]] = 1

        scenes.append({
            "dialogue_id": int(dialogue_id),
            "text": script,
            "label": label,
        })

    return scenes


def segment_scene(
    text: str,
    tokenizer,
    seg_size: int = MELD_CONFIG["SEG_SIZE"],
    shift: int = MELD_CONFIG["SHIFT"],
    max_length: int = MELD_CONFIG["MAX_LENGTH"],
) -> list[dict]:
    """
    씬 텍스트를 토큰 기반 슬라이딩 윈도우로 세그먼트 분할한다.

    Args:
        text: 씬 전체 텍스트
        tokenizer: HuggingFace 토크나이저
        seg_size: 세그먼트 크기 (CLS/SEP 제외 토큰 수)
        shift: 이동 폭
        max_length: 패딩 포함 최대 길이

    Returns:
        list of dict: [{"input_ids": list, "attention_mask": list}, ...]
    """
    if max_length < 3:
        raise ValueError("max_length는 특수 토큰 포함 최소 3 이상이어야 합니다.")

    # 전체 텍스트를 토큰화 (특수 토큰 제외)
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    if not all_tokens:
        all_tokens = tokenizer.encode(" ", add_special_tokens=False)

    segments = []
    start = 0
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)

    if cls_token_id is None:
        cls_token_id = getattr(tokenizer, "bos_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None:
        raise ValueError("토크나이저에 pad_token_id가 정의되어 있지 않습니다.")
    if cls_token_id is None or sep_token_id is None:
        raise ValueError("토크나이저에 CLS/SEP 또는 BOS/EOS 토큰 ID가 정의되어 있지 않습니다.")

    max_content_length = max_length - 2
    if max_content_length <= 0:
        raise ValueError("max_length는 특수 토큰 2개를 포함할 수 있어야 합니다.")

    while start < len(all_tokens):
        end = min(start + seg_size, len(all_tokens))
        seg_tokens = all_tokens[start:end][:max_content_length]

        # 구버전 transformers 호환을 위해 helper 없이 특수 토큰을 직접 붙인다.
        input_ids = [cls_token_id] + seg_tokens + [sep_token_id]
        attention_mask = [1] * len(input_ids)

        pad_length = max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        segments.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        # 이동: 마지막 세그먼트에 도달하면 종료
        if end >= len(all_tokens):
            break
        start += shift

    return segments


def preprocess_split(
    split: str,
    tokenizer,
) -> list[dict]:
    """
    하나의 split(train/dev/test)을 전처리한다.
    HuggingFace Datasets에서 자동으로 데이터를 가져온다.

    Args:
        split: "train", "dev", "test"
        tokenizer: HuggingFace 토크나이저

    Returns:
        list of dict: [{"dialogue_id", "label", "segments": [...], "num_segments"}, ...]
    """
    # HuggingFace에서 데이터 로드
    df = load_meld_hf(split)

    print(f"[전처리] {split} 씬 단위 변환 중... (발화 {len(df)}개)")
    scenes = build_scene_data(df)

    print(f"[전처리] {split} 세그먼테이션 중... (씬 {len(scenes)}개)")
    processed = []
    for scene in scenes:
        segments = segment_scene(scene["text"], tokenizer)
        processed.append({
            "dialogue_id": scene["dialogue_id"],
            "label": scene["label"],
            "segments": segments,
            "num_segments": len(segments),
        })

    return processed


def compute_segment_statistics(data: list[dict]) -> dict:
    """
    세그먼트 수 분포를 분석한다.
    FM 크기(r)와 CM 크기(k) 설정의 적절성을 검증하는 데 사용한다.

    Args:
        data: preprocess_split의 출력

    Returns:
        dict: 통계 정보 (min, max, mean, p95, 분포 등)
    """
    num_segs = [d["num_segments"] for d in data]
    num_segs = np.array(num_segs)

    fm_size = 3  # config의 r
    cm_size = 4  # config의 k

    stats = {
        "총 씬 수": len(num_segs),
        "세그먼트 수 최소": int(num_segs.min()),
        "세그먼트 수 최대": int(num_segs.max()),
        "세그먼트 수 평균": round(float(num_segs.mean()), 2),
        "세그먼트 수 중앙값": float(np.median(num_segs)),
        "세그먼트 수 p95": float(np.percentile(num_segs, 95)),
        "세그먼트 수 == 1 비율": round(float((num_segs == 1).mean()) * 100, 1),
        f"세그먼트 수 <= r({fm_size}) 비율": round(float((num_segs <= fm_size).mean()) * 100, 1),
        f"세그먼트 수 > r({fm_size}) 비율 (CM 활성)": round(float((num_segs > fm_size).mean()) * 100, 1),
        "세그먼트 수 분포": dict(sorted(Counter(num_segs.tolist()).items())),
    }

    return stats


def compute_pos_weight(data: list[dict], clip_max: float = 10.0) -> list[float]:
    """
    클래스별 pos_weight를 계산한다.
    BCEWithLogitsLoss에서 희귀 클래스에 높은 가중치를 부여한다.

    Args:
        data: preprocess_split의 출력
        clip_max: pos_weight 최댓값 제한

    Returns:
        list[float]: 10차원 pos_weight 벡터
    """
    labels = np.array([d["label"] for d in data])
    n = len(labels)
    pos_weight = []

    for i in range(labels.shape[1]):
        pos_count = labels[:, i].sum()
        neg_count = n - pos_count

        if pos_count == 0:
            # 양성 샘플이 없는 클래스 → 최대 가중치
            w = clip_max
        else:
            w = neg_count / pos_count

        pos_weight.append(min(float(w), clip_max))

    return pos_weight


def save_processed_data(data: list[dict], output_path: Path):
    """전처리된 데이터를 JSON으로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # numpy/torch 타입을 Python 기본 타입으로 변환
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=convert)

    print(f"[저장] {output_path} ({len(data)}개 씬)")


# ── 메인 실행 ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[설정] 출력 경로: {DATA_DIR}")
    print(f"[설정] 인코더: {ENCODER_CONFIG['MODEL_NAME']}")
    print(f"[설정] 세그먼트 크기: {MELD_CONFIG['SEG_SIZE']}, shift: {MELD_CONFIG['SHIFT']}")

    # 토크나이저 로드
    print("\n[토크나이저] 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])

    # 각 split 전처리 (HuggingFace에서 자동 다운로드)
    for split in ["train", "dev", "test"]:
        print(f"\n{'='*60}")
        data = preprocess_split(split, tokenizer)

        # 세그먼트 통계 출력
        stats = compute_segment_statistics(data)
        print(f"\n[통계] {split} 세그먼트 분포:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # 저장
        save_processed_data(data, DATA_DIR / f"{split}.json")

    # pos_weight 계산 (train 데이터 기준)
    print(f"\n{'='*60}")
    print("[pos_weight] train 데이터 기준으로 계산 중...")
    train_data = json.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
    from config import TRAIN_CONFIG
    pos_weight = compute_pos_weight(train_data, clip_max=TRAIN_CONFIG["POS_WEIGHT_CLIP"])

    all_labels = MELD_CONFIG["EMOTION_LABELS"] + MELD_CONFIG["SENTIMENT_LABELS"]
    print("\n[pos_weight] 클래스별 가중치:")
    for name, w in zip(all_labels, pos_weight):
        print(f"  {name}: {w:.2f}")

    # pos_weight 저장
    pw_path = DATA_DIR / "pos_weight.json"
    with open(pw_path, "w") as f:
        json.dump({"labels": all_labels, "pos_weight": pos_weight}, f, indent=2)
    print(f"\n[저장] {pw_path}")

    print("\n[완료] 전처리 완료!")
