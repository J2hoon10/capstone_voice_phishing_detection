"""
SNS 한국어 대화 전처리 모듈 (Mamba Belief 실험용)

Pipeline:
1. Data/Training/**/*.json 재귀 로드
2. info[0].annotations.subject 레이블 추출 및 정규화
3. 각 발화(line)를 개별 청크로 토크나이징 (KoELECTRA)
4. Training → train/val stratified split (90/10, seed=42)
5. Data/Validation/**/*.json → test
6. data/ 캐시에 train.json / val.json / test.json 저장

출력 포맷 (각 JSON 항목):
{
    "dialogue_id": str,
    "source_split": str,
    "source_path": str,
    "subject": str,
    "dialogue_label": int,   # 0~19 클래스 인덱스
    "num_turns": int,
    "turns": [{"input_ids": [...], "attention_mask": [...]}, ...]
}
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight
from transformers import AutoTokenizer

from config import DATA_DIR, DATA_ROOT, ENCODER_CONFIG, SNS_CONFIG


LABEL_TO_IDX = {label: idx for idx, label in enumerate(SNS_CONFIG["LABELS"])}


def normalize_subject(subject: Any) -> str:
    raw = str(subject).strip()
    return SNS_CONFIG["SUBJECT_NORMALIZATION"].get(raw, raw)


def safe_get_annotations(payload: dict) -> dict | None:
    info = payload.get("info")
    if not isinstance(info, list) or not info:
        return None
    first = info[0]
    if not isinstance(first, dict):
        return None
    annotations = first.get("annotations")
    if not isinstance(annotations, dict):
        return None
    return annotations


def extract_speaker_id(line: dict) -> str:
    speaker = line.get("speaker")
    if isinstance(speaker, dict):
        sid = speaker.get("id")
        if sid is not None and str(sid).strip():
            return str(sid).strip()

    text = str(line.get("text", "")).strip()
    if ":" in text:
        left = text.split(":", 1)[0].strip()
        if left:
            return left
    return "UNK"


def extract_norm_text(line: dict) -> str:
    norm_text = str(line.get("norm_text", "")).strip()
    if norm_text:
        return norm_text
    text = str(line.get("text", "")).strip()
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def build_turn_text(line: dict) -> str:
    speaker_id = extract_speaker_id(line)
    norm_text = extract_norm_text(line)
    return SNS_CONFIG["UTTERANCE_FORMAT"].format(
        speaker_id=speaker_id,
        norm_text=norm_text,
    )


def tokenize_lines(lines: list[dict], tokenizer) -> list[dict]:
    turns: list[dict] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = build_turn_text(line)
        encoded = tokenizer(
            text,
            max_length=SNS_CONFIG["MAX_LENGTH"],
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        turns.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return turns


def preprocess_json_file(
    file_path: Path, source_root: Path, source_split: str, tokenizer
) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None, "parse_error"

    annotations = safe_get_annotations(payload)
    if annotations is None:
        return None, "invalid_schema"

    subject = annotations.get("subject")
    if subject is None:
        return None, "missing_subject"

    subject_norm = normalize_subject(subject)
    if subject_norm not in LABEL_TO_IDX:
        return None, "unknown_subject"

    lines = annotations.get("lines")
    if not isinstance(lines, list) or len(lines) == 0:
        return None, "missing_lines"

    turns = tokenize_lines(lines, tokenizer)
    if not turns:
        return None, "empty_turns"

    rel_path = file_path.relative_to(source_root).as_posix()
    dialogue_id = f"{source_split}/{rel_path}"

    item = {
        "dialogue_id": dialogue_id,
        "source_split": source_split,
        "source_path": rel_path,
        "subject": subject_norm,
        "dialogue_label": LABEL_TO_IDX[subject_norm],
        "num_turns": len(turns),
        "turns": turns,
    }
    return item, None


def load_split_dialogues(
    source_root: Path, source_split: str, tokenizer
) -> tuple[list[dict], dict]:
    json_files = sorted(source_root.rglob("*.json"))
    processed: list[dict] = []
    skipped = Counter()

    for fp in json_files:
        item, err = preprocess_json_file(
            fp, source_root=source_root, source_split=source_split, tokenizer=tokenizer
        )
        if err is not None:
            skipped[err] += 1
            continue
        processed.append(item)

    info = {
        "source_split": source_split,
        "input_json_files": len(json_files),
        "processed_dialogues": len(processed),
        "skipped": dict(sorted(skipped.items())),
    }
    return processed, info


def build_stratified_train_val(
    training_dialogues: list[dict],
) -> tuple[list[dict], list[dict]]:
    labels = [d["dialogue_label"] for d in training_dialogues]
    indices = np.arange(len(training_dialogues))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=SNS_CONFIG["SPLIT_RATIOS"]["val"],
        random_state=SNS_CONFIG["SPLIT_SEED"],
        shuffle=True,
        stratify=labels,
    )
    train_data = [training_dialogues[i] for i in train_idx]
    val_data = [training_dialogues[i] for i in val_idx]
    return train_data, val_data


def split_distribution(dialogues: list[dict]) -> dict[str, int]:
    counts = Counter(d["subject"] for d in dialogues)
    return {label: counts.get(label, 0) for label in SNS_CONFIG["LABELS"]}


def compute_class_weights(train_dialogues: list[dict]) -> dict:
    y = np.array([d["dialogue_label"] for d in train_dialogues], dtype=np.int64)
    classes = np.arange(SNS_CONFIG["NUM_LABELS"], dtype=np.int64)
    counts = Counter(y.tolist())

    if y.size == 0:
        class_weight = np.ones(SNS_CONFIG["NUM_LABELS"], dtype=np.float64)
        focal_alpha = np.ones(SNS_CONFIG["NUM_LABELS"], dtype=np.float64)
    else:
        class_weight = sk_compute_class_weight(
            class_weight="balanced", classes=classes, y=y
        )
        beta = 0.999
        cnt = np.array([counts.get(i, 0) for i in classes], dtype=np.float64)
        focal_alpha = np.zeros_like(cnt, dtype=np.float64)
        nonzero = cnt > 0
        focal_alpha[nonzero] = (1.0 - beta) / (
            1.0 - np.power(beta, cnt[nonzero])
        )
        if nonzero.any():
            focal_alpha = focal_alpha / focal_alpha[nonzero].sum() * nonzero.sum()
        else:
            focal_alpha[:] = 1.0

    return {
        "labels": SNS_CONFIG["LABELS"],
        "class_count": [int(counts.get(i, 0)) for i in classes],
        "class_weight": [round(float(x), 6) for x in class_weight.tolist()],
        "focal_alpha": [round(float(x), 6) for x in focal_alpha.tolist()],
        "focal_beta": 0.999,
    }


def save_json(data: Any, path: Path, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)


def print_distribution(name: str, dialogues: list[dict]) -> None:
    dist = split_distribution(dialogues)
    total = len(dialogues)
    print(f"\n[{name}] {total} dialogues")
    for label in SNS_CONFIG["LABELS"]:
        count = dist[label]
        ratio = (count / total * 100.0) if total else 0.0
        print(f"  - {label}: {count} ({ratio:.2f}%)")


def main() -> None:
    train_root = DATA_ROOT / "Training"
    test_root = DATA_ROOT / "Validation"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Raw data path missing. Expected:\n- {train_root}\n- {test_root}"
        )

    print(f"[config] DATA_ROOT: {DATA_ROOT}")
    print(f"[config] OUTPUT_DIR: {DATA_DIR}")
    print(f"[config] Encoder: {ENCODER_CONFIG['MODEL_NAME']}")
    print(f"[config] Max length: {SNS_CONFIG['MAX_LENGTH']}")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])

    print("\n[load] Training split preprocessing...")
    raw_training, train_info = load_split_dialogues(
        train_root, source_split="training", tokenizer=tokenizer
    )
    print(f"[load] Training input JSON: {train_info['input_json_files']}")
    print(f"[load] Training processed: {train_info['processed_dialogues']}")
    if train_info["skipped"]:
        print(f"[load] Training skipped: {train_info['skipped']}")

    train_data, val_data = build_stratified_train_val(raw_training)

    print("\n[load] Validation(Test) split preprocessing...")
    test_data, test_info = load_split_dialogues(
        test_root, source_split="validation", tokenizer=tokenizer
    )
    print(f"[load] Validation input JSON: {test_info['input_json_files']}")
    print(f"[load] Validation processed: {test_info['processed_dialogues']}")
    if test_info["skipped"]:
        print(f"[load] Validation skipped: {test_info['skipped']}")

    print_distribution("train", train_data)
    print_distribution("val", val_data)
    print_distribution("test", test_data)

    save_json(train_data, DATA_DIR / "train.json")
    save_json(val_data, DATA_DIR / "val.json")
    save_json(test_data, DATA_DIR / "test.json")

    label_map = {label: idx for idx, label in enumerate(SNS_CONFIG["LABELS"])}
    save_json(label_map, DATA_DIR / "label_map.json", pretty=True)

    class_weight_data = compute_class_weights(train_data)
    save_json(class_weight_data, DATA_DIR / "class_weight.json", pretty=True)

    meta = {
        "num_labels": SNS_CONFIG["NUM_LABELS"],
        "split_ratios": SNS_CONFIG["SPLIT_RATIOS"],
        "seed": SNS_CONFIG["SPLIT_SEED"],
        "training_summary": train_info,
        "validation_summary": test_info,
        "output_counts": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
    }
    save_json(meta, DATA_DIR / "preprocess_meta.json", pretty=True)

    print("\n[save] train.json / val.json / test.json")
    print("[save] label_map.json / class_weight.json / preprocess_meta.json")
    print("[done] preprocessing completed")


if __name__ == "__main__":
    main()
