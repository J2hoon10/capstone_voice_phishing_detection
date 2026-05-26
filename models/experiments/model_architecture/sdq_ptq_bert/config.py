"""
SDQ-LLM PTQ experiment configuration.

Target task:
- NSMC (Naver Sentiment Movie Corpus) binary sentiment classification
- Encoder: klue/bert-base
- Quantization: Sigma-Delta Quantization (SDQ-LLM, arXiv 2510.03275)
"""

from pathlib import Path

import torch

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimizations
GPU_CONFIG = {
    "ENABLED": True,
    "DTYPE": "bf16",
    "TF32_MATMUL": True,
    "TF32_CUDNN": True,
    "TARGET_VRAM_GB": 14.0,
    "CUDNN_BENCHMARK": True,
    "PIN_MEMORY": True,
    "NUM_WORKERS": 0,
}

# NSMC dataset config
DATA_CONFIG = {
    "DATASET_NAME": "nsmc",
    "NUM_LABELS": 2,
    "LABELS": ["negative", "positive"],
    "MAX_LENGTH": 128,
    "TEXT_COLUMN": "document",
    "LABEL_COLUMN": "label",
}

# Encoder
ENCODER_CONFIG = {
    "MODEL_NAME": "klue/bert-base",
    "HIDDEN_SIZE": 768,
    "NUM_LAYERS": 12,
    "NUM_HEADS": 12,
}

# Fine-tuning
TRAIN_CONFIG = {
    "BATCH_SIZE": 32,
    "EPOCHS": 3,
    "SEED": 42,
    "LR": 2e-5,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MAX_GRAD_NORM": 1.0,
    "VAL_RATIO": 0.1,
}

# Evaluation
EVAL_CONFIG = {
    "BATCH_SIZE": 64,
    "NUM_WARMUP_BATCHES": 10,
}

# Quantization
QUANT_CONFIG = {
    # Bit-widths to test
    "BIT_WIDTHS": [1, 1.58, 2],
    # OSR values per bit-width
    "OSR_MAP": {
        1: [2, 4, 8, 16, 32],
        1.58: [2, 4, 8, 16],
        2: [2, 4, 8],
    },
    # Adaptive quantization target average bit-widths
    "ADAPTIVE_TARGETS": [1.0, 1.5, 2.0],
    # Layer name patterns to skip (keep in FP32)
    "SKIP_PATTERNS": [
        "embeddings",
        "LayerNorm",
        "layernorm",
        "classifier",
        "pooler",
    ],
    # Sigma-delta modulator order
    "SD_ORDER": 1,
}
