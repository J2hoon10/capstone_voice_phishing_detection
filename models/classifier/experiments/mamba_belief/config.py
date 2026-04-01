"""
Mamba Belief 실험 설정 파일

RoBERTa(청크 인코더) + Mamba SSM(청크 간 Belief 누적) 기반
MELD 스트리밍 분류 모델의 하이퍼파라미터 및 경로 설정.

데이터는 compressive_memory 실험에서 전처리된 것을 그대로 재사용한다.
"""

from pathlib import Path

import torch

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent                              # experiments/mamba_belief/
PROJECT_ROOT = BASE_DIR.parents[2]                                       # capstone_voice_phishing_detection/
DATA_DIR = BASE_DIR.parent / "compressive_memory" / "data"              # 기존 전처리 데이터 재사용
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# ── 디바이스 설정 ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── GPU 최적화 설정 (RTX 4070 Ti Super, 16GB VRAM) ────────
GPU_CONFIG = {
    "ENABLED": True,
    "DTYPE": "bf16",
    "TF32_MATMUL": True,
    "TF32_CUDNN": True,
    "TARGET_VRAM_GB": 14.0,
    "GRADIENT_ACCUMULATION_STEPS": 2,
    "EMPTY_CACHE_INTERVAL": 0,
    "CUDNN_BENCHMARK": True,
    "PIN_MEMORY": True,
    "NUM_WORKERS": 2,
}

# ── MELD 데이터셋 설정 ────────────────────────────────────
MELD_CONFIG = {
    "EMOTION_LABELS": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "SENTIMENT_LABELS": ["positive", "negative", "neutral"],
    "NUM_LABELS": 10,
    "SEG_SIZE": 200,
    "SHIFT": 50,
    "MAX_LENGTH": 512,
    "UTTERANCE_FORMAT": "{speaker}: {utterance}",
    "UTTERANCE_SEP": " ",
}

# ── 인코더 설정 ────────────────────────────────────────────
ENCODER_CONFIG = {
    "MODEL_NAME": "roberta-base",
    "HIDDEN_SIZE": 768,
    "PAD_TOKEN_ID": 1,
    "FREEZE_ENCODER": True,   # Phase 1: 인코더 고정, Mamba+Head만 학습
}

# ── Mamba SSM 설정 ─────────────────────────────────────────
MAMBA_CONFIG = {
    "D_MODEL": 768,      # BERT hidden size와 동일 (projection 불필요)
    "D_STATE": 16,       # SSM state 차원 (ablation: 16/32/64)
    "D_CONV": 4,         # Mamba 내부 Conv1d 커널 크기
    "EXPAND": 2,         # inner projection 확장 비율 (d_inner = EXPAND * D_MODEL)
    "NUM_LAYERS": 1,     # Mamba 레이어 수 (ablation: 1/2)
}

# ── Classification Head 설정 ───────────────────────────────
HEAD_CONFIG = {
    "HIDDEN_DIM": 64,
    "DROPOUT": 0.1,
}

# ── 학습 설정 ──────────────────────────────────────────────
TRAIN_CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 20,
    "SEED": 42,
    "ENCODER_LR": 2e-5,
    "UPPER_LR": 5e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MIN_LR": 1e-6,
    "PATIENCE": 7,
    "MIN_DELTA": 1e-4,
    "LOSS_STRATEGY": "L3",
    "POS_WEIGHT_CLIP": 10.0,
    "MAX_GRAD_NORM": 1.0,
}

# ── 평가 설정 ──────────────────────────────────────────────
EVAL_CONFIG = {
    "THRESHOLD": 0.5,
    "THRESHOLD_SEARCH_RANGE": (0.2, 0.8, 0.05),
}

# ── Ablation 실험 설정 ─────────────────────────────────────
ABLATION_CONFIGS = {
    # A: Baseline (Mamba 없음 — CLS 토큰 직접 분류)
    "baseline": {
        "SKIP_MAMBA": True,
        "FREEZE_ENCODER": True,
        "description": "Mamba 없음, 각 세그먼트 CLS 토큰으로만 분류",
    },
    # B: Mamba + 인코더 고정 (권장 시작점)
    "mamba_frozen": {
        "FREEZE_ENCODER": True,
        "D_STATE": 16,
        "description": "RoBERTa 고정 + Mamba SSM(d_state=16)",
    },
    # C: Mamba + 인코더 fine-tune
    "mamba_finetune": {
        "FREEZE_ENCODER": False,
        "D_STATE": 16,
        "description": "RoBERTa fine-tune + Mamba SSM(d_state=16)",
    },
    # D: SSM state 차원 ablation
    "mamba_state32": {
        "FREEZE_ENCODER": True,
        "D_STATE": 32,
        "description": "Mamba SSM d_state=32",
    },
    "mamba_state64": {
        "FREEZE_ENCODER": True,
        "D_STATE": 64,
        "description": "Mamba SSM d_state=64",
    },
    # E: Mamba 레이어 수 ablation
    "mamba_2layers": {
        "FREEZE_ENCODER": True,
        "D_STATE": 16,
        "NUM_LAYERS": 2,
        "description": "Mamba 레이어 2개 스택",
    },
}
