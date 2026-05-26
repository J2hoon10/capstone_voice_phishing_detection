from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "models" / "main" / "data_augmentation" / "output" / "4class"
STN_LABELING_DIR = PROJECT_ROOT / "models" / "remain" / "stn_labeling" / "output"
# map_segments.py --window-size 64 --stride 32 로 생성 (기존 segment_risks.csv 보존)
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

# 계층적 클래스 구조
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
    "MODEL_NAME": "monologg/koelectra-base-v3-discriminator",
}

MAMBA_CONFIG = {
    "D_MODEL": 768,
    "D_STATE": 16,
    "D_CONV": 4,
    "EXPAND": 2,
    "NUM_LAYERS": 1,
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
    "ENCODER_LR": 2e-5,
    "UPPER_LR": 5e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MIN_LR": 1e-6,
    "MAX_GRAD_NORM": 1.0,
    # freeze_init: 처음 N 에폭은 encoder 완전 freeze → Mamba+head만 먼저 수렴
    # N+1 에폭부터 encoder 상위 레이어(layer[10~11]) unfreeze하여 fine-tune
    "FREEZE_INIT_EPOCHS": 2,
}
