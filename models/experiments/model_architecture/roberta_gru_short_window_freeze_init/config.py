from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "models" / "main" / "data_augmentation" / "output" / "4class"
STN_LABELING_DIR = PROJECT_ROOT / "models" / "remain" / "stn_labeling" / "output"
SEGMENT_RISKS_CSV = STN_LABELING_DIR / "segment_risks_w64_s32.csv"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "상품 가입 및 해지",
    "이체 출금 대출서비스",
    "잔고 및 거래내역",
    "수사기관 사칭형",
    "대출 사기형",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}
NUM_LABELS = len(LABELS)

SUPERCLASS_MAP = {
    0: 0,  # 상품 가입 및 해지   -> 일반
    1: 0,  # 이체 출금 대출서비스 -> 일반
    2: 0,  # 잔고 및 거래내역    -> 일반
    3: 1,  # 수사기관 사칭형     -> 피싱
    4: 1,  # 대출 사기형        -> 피싱
}
NUM_SUPERCLASSES = 2

WINDOW_SIZE = 64
STRIDE = 32

ENCODER_CONFIG = {
    "MODEL_NAME": "klue/roberta-base",
}

RNN_CONFIG = {
    "D_MODEL": 768,      # 인코더 출력 차원 (= GRU input_size)
    "HIDDEN_SIZE": 768,  # GRU hidden_size (unidirectional → 출력 차원 동일)
    "NUM_LAYERS": 2,
    "DROPOUT": 0.1,      # GRU inter-layer dropout (num_layers > 1 일 때 적용)
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
    "ENCODER_LR": 2e-5,
    "UPPER_LR": 5e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MIN_LR": 1e-6,
    "MAX_GRAD_NORM": 1.0,
    "FREEZE_INIT_EPOCHS": 2,
}
