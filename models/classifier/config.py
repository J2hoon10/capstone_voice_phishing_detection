import os
from pathlib import Path

import torch


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "weights" / "student_best.pt"

CONFIG = {
    "DEVICE": os.getenv("CLASSIFIER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    "BASE_MODEL_NAME": os.getenv("BASE_MODEL_NAME", "neavo/modern_bert_multilingual"),
    "NUM_LABELS": _get_env_int("NUM_LABELS", 2),
    "MAX_LENGTH": _get_env_int("MAX_LENGTH", 1024),
    "WINDOW_SIZE": _get_env_int("WINDOW_SIZE", 15),
    "STRIDE": _get_env_int("STRIDE", 5),
    "DEFAULT_THRESHOLD": _get_env_float("DEFAULT_THRESHOLD", 0.5),
    "MODEL_PATH": os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    "WHISPER_MODEL_SIZE": os.getenv("WHISPER_MODEL_SIZE", "Systran/faster-whisper-base"),
    "STUDENT": {
        "config": {
            "hidden_size": _get_env_int("STUDENT_HIDDEN_SIZE", 384),
            "num_hidden_layers": _get_env_int("STUDENT_NUM_HIDDEN_LAYERS", 11),
            "num_attention_heads": _get_env_int("STUDENT_NUM_ATTENTION_HEADS", 6),
            "intermediate_size": _get_env_int("STUDENT_INTERMEDIATE_SIZE", 576),
        }
    },
}

