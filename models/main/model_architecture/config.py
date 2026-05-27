import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2] # capstone_voice_phishing_detection

# Data 및 logs 경로
DATA_DIR = PROJECT_ROOT / "models" / "main" / "data_augmentation" / "output" / "4class"
STN_LABELING_DIR = PROJECT_ROOT / "models" / "remain" / "stn_labeling" / "output"
SEGMENT_RISKS_CSV = STN_LABELING_DIR / "segment_risks.csv"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "상담 대화",
    "일상 대화",
    "대출 사기형",
    "수사기관 사칭형",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}
NUM_LABELS = len(LABELS)

# 계층적 클래스 구조: 0~1 일반(superclass 0), 2~3 피싱(superclass 1)
SUPERCLASS_MAP = {
    0: 0,  # 상담 대화   -> 일반
    1: 0,  # 일상 대화   -> 일반
    2: 1,  # 대출 사기형  -> 피싱
    3: 1,  # 수사기관 사칭형 -> 피싱
}
NUM_SUPERCLASSES = 2

# 표준 고정 윈도우
WINDOW_SIZE = 64      # 슬라이딩 윈도우 목표 크기 (content = 62 tokens)
STRIDE = 32
MAX_SEQ_LEN = 128     # 실제 패딩 기준

ENCODER_CONFIG = {
    "MODEL_NAME": "klue/roberta-base",
}

MAMBA_CONFIG = {
    "D_MODEL": 768,
    "D_STATE": 16,
    "D_CONV": 4,
    "EXPAND": 2,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.1,
}

HEAD_CONFIG = {
    "HIDDEN_DIM": 64,
    "DROPOUT": 0.1,
}

LOSS_CONFIG = {
    "HCE_LAMBDA": 0.5,
    "AUX_BETA_START": 0.5,
    "AUX_BETA_END": 0.1,
    "FOCAL_GAMMA": 2.0,
}

TRAIN_CONFIG = {
    "BATCH_SIZE": 4,
    "EPOCHS": 8,
    "SEED": 42,
    "ENCODER_LR": 1e-5,
    "UPPER_LR": 5e-4,
    "UPPER_MIN_LR": 1e-4,
    "ENCODER_MIN_LR": 1e-6,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MAX_GRAD_NORM": 1.0,
    "FREEZE_INIT_EPOCHS": 3,
}

# FastAPI 및 추론기 서비스 설정
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


DEFAULT_MODEL_PATH = BASE_DIR / "weights" / "student_best.pt"

CONFIG = {
    "DEVICE": os.getenv("CLASSIFIER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
    "MODEL_KIND": os.getenv("CLASSIFIER_MODEL_KIND", "roberta_mamba"),
    "BASE_MODEL_NAME": os.getenv("BASE_MODEL_NAME", "klue/roberta-base"),
    "NUM_LABELS": _get_env_int("NUM_LABELS", 4),
    "MAX_LENGTH": _get_env_int("MAX_LENGTH", 128),
    "WINDOW_SIZE": _get_env_int("WINDOW_SIZE", 64),
    "STRIDE": _get_env_int("STRIDE", 32),
    "DEFAULT_THRESHOLD": _get_env_float("DEFAULT_THRESHOLD", 0.5),
    "MODEL_PATH": os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)),
    "WHISPER_MODEL_SIZE": os.getenv("WHISPER_MODEL_SIZE", "Systran/faster-whisper-base"),
}
