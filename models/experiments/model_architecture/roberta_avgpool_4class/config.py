from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "models" / "classifier" / "preprocessing" / "output" / "4class"
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

WINDOW_SIZE = 64      # 슬라이딩 윈도우 목표 크기 (content = 62 tokens)
STRIDE = 32
MAX_SEQ_LEN = 128     # 실제 패딩 기준 (단어/문장 경계 확장 여유분 포함)

ENCODER_CONFIG = {
    "MODEL_NAME": "klue/roberta-base",
}

HEAD_CONFIG = {
    "HIDDEN_DIM": 64,
    "DROPOUT": 0.1,
}

TRAIN_CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 8,
    "SEED": 42,
    "ENCODER_LR": 2e-5,
    "UPPER_LR": 5e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MIN_LR": 1e-6,
    "MAX_GRAD_NORM": 1.0,
}
