"""
Compressive Memory KR experiment configuration.

Target task:
- SNS multi-turn dialogue-level subject classification (20 classes)
- Encoder: monologg/koelectra-base-v3-discriminator
"""

from pathlib import Path

import torch


# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
DATA_ROOT = PROJECT_ROOT / "normal_data_reference"  # raw JSON root (Training/Validation)
DATA_DIR = BASE_DIR / "data"  # preprocessed cache files (train/val/test)
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GPU optimizations
GPU_CONFIG = {
    "ENABLED": True,
    "DTYPE": "bf16",  # "bf16" | "fp16"
    "TF32_MATMUL": True,
    "TF32_CUDNN": True,
    "TARGET_VRAM_GB": 14.0,
    "GRADIENT_ACCUMULATION_STEPS": 2,
    "EMPTY_CACHE_INTERVAL": 0,
    "CUDNN_BENCHMARK": True,
    "PIN_MEMORY": True,
    "NUM_WORKERS": 0,
}


# SNS dialogue-level classification config
SNS_CONFIG = {
    "LABELS": [
        "가족",
        "건강",
        "게임",
        "계절/날씨",
        "교육",
        "교통",
        "군대",
        "미용",
        "반려동물",
        "방송/연예",
        "사회이슈",
        "상거래 전반",
        "스포츠/레저",
        "식음료",
        "여행",
        "영화/만화",
        "연애/결혼",
        "주거와 생활",
        "타 국가 이슈",
        "회사/아르바이트",
    ],
    "NUM_LABELS": 20,
    "MAX_LENGTH": 128,
    "UTTERANCE_FORMAT": "{speaker_id}: {norm_text}",
    "SPLIT_RATIOS": {"train": 0.9, "val": 0.1},
    "SPLIT_SEED": 42,
    "SUBJECT_NORMALIZATION": {
        "상거래전반": "상거래 전반",
    },
}


# Encoder
ENCODER_CONFIG = {
    "MODEL_NAME": "monologg/koelectra-base-v3-discriminator",
    "HIDDEN_SIZE": 768,
    "PAD_TOKEN_ID": 0,
    "FREEZE_ENCODER": False,
}


# Compressive Memory
MEMORY_CONFIG = {
    "SLOT_DIM": 128,
    "FM_SIZE": 3,
    "CM_SIZE": 4,
    "CONV_KERNEL_SIZE": 2,
    "USE_LEARNABLE_INIT": True,
}


# Attention (kept for compatibility, used inside model components)
ATTENTION_CONFIG = {
    "NUM_HEADS": 8,
    "DROPOUT": 0.1,
}


# Classification head
HEAD_CONFIG = {
    "HIDDEN_DIM": 64,
    "DROPOUT": 0.1,
}


# Training
TRAIN_CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 5,
    "SEED": 42,
    "ENCODER_LR": 1e-5,
    "UPPER_LR": 5e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MIN_LR": 1e-6,
    "PATIENCE": 7,
    "MIN_DELTA": 1e-4,
    "LOSS_FN": "cross_entropy",  # "cross_entropy" | "focal" | "weighted_ce"
    "FOCAL_GAMMA": 2.0,
    "MAX_GRAD_NORM": 1.0,
}


EVAL_CONFIG = {
    "TOP_K": 1,
}


# Ablations
ABLATION_CONFIGS = {
    "baseline": {
        "FM_SIZE": 0,
        "CM_SIZE": 0,
        "description": "No memory; mean pooled turn encoding baseline",
    },
    "fm_only": {
        "FM_SIZE": 3,
        "CM_SIZE": 0,
        "description": "Fine memory only",
    },
    "full": {
        "FM_SIZE": 3,
        "CM_SIZE": 4,
        "description": "FM + CM full model",
    },
    "mean_pooling": {
        "FM_SIZE": 3,
        "CM_SIZE": 4,
        "COMPRESS_FN": "mean",
        "description": "Use mean compression instead of conv",
    },
    "cm_k1": {"CM_SIZE": 1, "description": "CM slots k=1"},
    "cm_k2": {"CM_SIZE": 2, "description": "CM slots k=2"},
    "cm_k8": {"CM_SIZE": 8, "description": "CM slots k=8"},
}

