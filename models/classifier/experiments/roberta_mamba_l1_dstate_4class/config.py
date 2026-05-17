from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "models" / "classifier" / "preprocessing" / "output" / "4class"
STN_LABELING_DIR = PROJECT_ROOT / "models" / "classifier" / "preprocessing" / "stn_labeling" / "output"
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

# 계층적 클래스 구조
SUPERCLASS_MAP = {
    0: 0,  # 상담 대화      -> 일반
    1: 0,  # 일상 대화      -> 일반
    2: 1,  # 대출 사기형    -> 피싱
    3: 1,  # 수사기관 사칭형 -> 피싱
}
NUM_SUPERCLASSES = 2

WINDOW_SIZE = 64
STRIDE = 32
MAX_SEQ_LEN = 128

ENCODER_CONFIG = {
    "MODEL_NAME": "klue/roberta-base",
}

# D_STATE는 train.py / evaluate.py 에서 --d-state 인수로 런타임 오버라이드 가능
MAMBA_CONFIG = {
    "D_MODEL": 768,
    "D_STATE": 16,   # 기본값 — 실험 변수: 16 / 32 / 64
    "D_CONV": 4,
    "EXPAND": 2,
    "NUM_LAYERS": 1,  # roberta_mamba_freeze_init_4class(L2) 대비 L1 고정
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
