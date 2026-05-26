from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "models" / "classifier" / "preprocessing" / "output" / "4class_w32"
STN_LABELING_DIR = PROJECT_ROOT / "models" / "classifier" / "preprocessing" / "stn_labeling" / "output"
# 표준 window (128/100) 로 생성된 segment_risks (기본 파일명)
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

# 계층적 클래스 구조: 0~2 일반(superclass 0), 3~4 피싱(superclass 1)
SUPERCLASS_MAP = {
    0: 0,  # 상담 대화   -> 일반
    1: 0,  # 일상 대화   -> 일반
    2: 1,  # 대출 사기형  -> 피싱
    3: 1,  # 수사기관 사칭형 -> 피싱
}
NUM_SUPERCLASSES = 2

# 짧은 윈도우 실험 (roberta_mamba_freeze_init_4class 대비 윈도우 절반)
WINDOW_SIZE = 32      # 슬라이딩 윈도우 목표 크기 (content = 30 tokens)
STRIDE = 16
MAX_SEQ_LEN = 64      # 실제 패딩 기준 (단어/문장 경계 확장 여유분 포함)

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
    # RoBERTa 학습률 (short 버전 2e-5 대비 더 보수적)
    "ENCODER_LR": 1e-5,
    # Mamba+Head 초기 학습률 → cosine decay → UPPER_MIN_LR
    "UPPER_LR": 5e-4,
    "UPPER_MIN_LR": 1e-4,    # Mamba+Head cosine 하한 (short 버전: 1e-6)
    "ENCODER_MIN_LR": 1e-6,  # encoder cosine 하한
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MAX_GRAD_NORM": 1.0,
    # freeze_init: 처음 N 에폭은 encoder 완전 freeze → Mamba+head 만 수렴
    # N+1 에폭부터 encoder layer[10~11] unfreeze → fine-tune
    "FREEZE_INIT_EPOCHS": 3,  # short 버전: 2
}
