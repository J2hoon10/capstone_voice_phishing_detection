from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "models" / "classifier" / "preprocessing" / "final"
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

# 계층적 클래스 구조 정의
# 0~2: 일반(superclass 0), 3~4: 피싱(superclass 1)
SUPERCLASS_MAP = {
    0: 0,  # 상품 가입 및 해지   -> 일반
    1: 0,  # 이체 출금 대출서비스 -> 일반
    2: 0,  # 잔고 및 거래내역    -> 일반
    3: 1,  # 수사기관 사칭형     -> 피싱
    4: 1,  # 대출 사기형        -> 피싱
}
NUM_SUPERCLASSES = 2  # 일반 / 피싱

WINDOW_SIZE = 128
STRIDE = 100

ENCODER_CONFIG = {
    "MODEL_NAME": "monologg/koelectra-base-v3-discriminator",
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
    # Hierarchical Cross-Entropy: subclass 손실 가중치 (spec 3.2의 λ)
    "HCE_LAMBDA": 0.5,
    # Ordinal aux loss 가중치 α·L_main + β·L_aux 에서 β
    # Curriculum: 초기 β_start → 최종 β_end 로 선형 감소
    "AUX_BETA_START": 0.5,
    "AUX_BETA_END": 0.1,
    # Focal Loss: subclass 손실에 적용할 감마 (0 이면 일반 CE)
    "FOCAL_GAMMA": 2.0,
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
