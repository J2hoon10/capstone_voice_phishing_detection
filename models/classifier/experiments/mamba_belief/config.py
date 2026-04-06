"""
Mamba Belief 실험 설정 파일

KoELECTRA(청크 인코더) + Mamba SSM(청크 간 Belief 누적) 기반
SNS 한국어 대화 스트리밍 분류 모델의 하이퍼파라미터 및 경로 설정.

데이터셋: 한국어 SNS 일상대화 (Data/Training, Data/Validation)
태스크: 20-class single-label subject 분류
"""

from pathlib import Path

import torch

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent                              # experiments/mamba_belief/
PROJECT_ROOT = BASE_DIR.parents[3]                                       # capstone_voice_phishing_detection/
DATA_ROOT = PROJECT_ROOT / "Data"                                        # 원본 JSON (Training/ Validation/)
DATA_DIR = BASE_DIR / "data"                                             # 전처리된 캐시 (train/val/test)
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
    "NUM_WORKERS": 0,
}

# ── SNS 한국어 데이터셋 설정 ──────────────────────────────
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

# ── 인코더 설정 ────────────────────────────────────────────
ENCODER_CONFIG = {
    "MODEL_NAME": "monologg/koelectra-base-v3-discriminator",
    "HIDDEN_SIZE": 768,
    "PAD_TOKEN_ID": 0,   # KoELECTRA pad token id
    "FREEZE_ENCODER": True,   # Phase 1: 인코더 고정, Mamba+Head만 학습
}

# ── Mamba SSM 설정 ─────────────────────────────────────────
MAMBA_CONFIG = {
    "D_MODEL": 768,      # BERT hidden size와 동일 (projection 불필요)
    "D_STATE": 16,       # SSM state 차원 (ablation: 16/32/64)
    "D_CONV": 4,         # Mamba 내부 Conv1d 커널 크기
    "EXPAND": 2,         # inner projection 확장 비율 (d_inner = EXPAND * D_MODEL)
    "NUM_LAYERS": 3,     # Mamba 레이어 수 (ablation: 1/2)
}

# ── Classification Head 설정 ───────────────────────────────
HEAD_CONFIG = {
    "HIDDEN_DIM": 64,
    "DROPOUT": 0.1,
}

# ── 학습 설정 ──────────────────────────────────────────────
TRAIN_CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 5,
    "SEED": 42,
    "ENCODER_LR": 2e-5,
    "UPPER_LR": 5e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MIN_LR": 1e-6,
    "PATIENCE": 3,
    "MIN_DELTA": 1e-4,
    "LOSS_STRATEGY": "L3",   # "L1": 마지막만, "L2": 균등, "L3": 시간 비례 가중
    "MAX_GRAD_NORM": 1.0,
}

# ── 평가 설정 ──────────────────────────────────────────────
EVAL_CONFIG = {
    "TOP_K": 1,
    "ENTROPY_THRESHOLD": 0.3,  # 조기 종료 엔트로피 임계값 (log20 ≈ 3.0, 정규화 기준)
}

# ── Ablation 실험 설정 ─────────────────────────────────────
ABLATION_CONFIGS = {
    # A: Baseline (Mamba 없음 — CLS 토큰 직접 분류)
    "baseline": {
        "SKIP_MAMBA": True,
        "FREEZE_ENCODER": False,
        "description": "Mamba 없음, 각 청크 CLS 토큰으로만 분류",
    },
    # B: Mamba + 인코더 고정 (권장 시작점)
    "mamba_frozen": {
        "FREEZE_ENCODER": True,
        "D_STATE": 16,
        "description": "KoELECTRA 고정 + Mamba SSM(d_state=16)",
    },
    # C: Mamba + 인코더 fine-tune
    "mamba_finetune": {
        "FREEZE_ENCODER": False,
        "D_STATE": 16,
        "description": "KoELECTRA fine-tune + Mamba SSM(d_state=16)",
    },
    # D: SSM state 차원 ablation
    "mamba_state32": {
        "FREEZE_ENCODER": False,
        "D_STATE": 32,
        "description": "Mamba SSM d_state=32",
    },
    "mamba_state64": {
        "FREEZE_ENCODER": False,
        "D_STATE": 64,
        "description": "Mamba SSM d_state=64",
    },
    # E: Mamba 레이어 수 ablation
    "mamba_2layers": {
        "FREEZE_ENCODER": False,
        "D_STATE": 16,
        "NUM_LAYERS": 2,
        "description": "Mamba 레이어 2개 스택",
    },
}
