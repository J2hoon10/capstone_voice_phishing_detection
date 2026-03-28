"""
Compressive Memory 실험 설정 파일

MELD 데이터셋을 활용한 Compressive Memory 기반 스트리밍 분류 모델의
하이퍼파라미터 및 경로 설정을 관리한다.
"""

import os
from pathlib import Path

import torch

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent                    # experiments/compressive_memory/
PROJECT_ROOT = BASE_DIR.parents[2]                            # capstone_voice_phishing_detection/
DATA_DIR = BASE_DIR / "data"                                  # 전처리된 데이터 저장 경로
CHECKPOINT_DIR = BASE_DIR / "checkpoints"                     # 모델 체크포인트 저장 경로
LOG_DIR = BASE_DIR / "logs"                                   # 학습 로그 저장 경로

# ── 디바이스 설정 ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── GPU 최적화 설정 (RTX 4070 Ti Super, 16GB VRAM) ────────
GPU_CONFIG = {
    # Mixed Precision 설정
    "ENABLED": True,                    # AMP(Automatic Mixed Precision) 사용 여부
    "DTYPE": "bf16",                    # "bf16" 또는 "fp16" (Ada Lovelace는 bf16 네이티브 지원)
    "TF32_MATMUL": True,               # TF32 행렬 연산 활성화 (fp32 연산 가속)
    "TF32_CUDNN": True,                # cuDNN TF32 활성화

    # VRAM 관리
    "TARGET_VRAM_GB": 14.0,            # 목표 VRAM 사용량 (16GB - 2GB 여유)
    "GRADIENT_ACCUMULATION_STEPS": 2,  # 유효 배치 크기 = BATCH_SIZE × 이 값
    "EMPTY_CACHE_INTERVAL": 0,         # N 배치마다 캐시 비우기 (0이면 비활성)

    # 성능 최적화
    "CUDNN_BENCHMARK": True,           # 입력 크기 고정 시 cuDNN 자동 최적 알고리즘 탐색
    "PIN_MEMORY": True,                # DataLoader에서 pin_memory 사용
    "NUM_WORKERS": 2,                  # DataLoader worker 수
}

# ── MELD 데이터셋 설정 ────────────────────────────────────
MELD_CONFIG = {
    # 레이블 정의
    "EMOTION_LABELS": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "SENTIMENT_LABELS": ["positive", "negative", "neutral"],
    "NUM_LABELS": 10,  # 감정 7종 + 감성 3종 = 10차원 multi-hot

    # 세그먼테이션 파라미터
    "SEG_SIZE": 200,        # 세그먼트 크기 (토큰 수, CLS/SEP 제외)
    "SHIFT": 50,            # 이동 폭 (인접 세그먼트 150토큰 중첩)
    "MAX_LENGTH": 512,      # 패딩 포함 최대 길이

    # 씬 텍스트 생성 포맷
    "UTTERANCE_FORMAT": "{speaker}: {utterance}",  # "Speaker: Utterance" 형식
    "UTTERANCE_SEP": " ",                          # 발화 간 구분자
}

# ── 인코더 설정 ────────────────────────────────────────────
ENCODER_CONFIG = {
    "MODEL_NAME": "roberta-base",       # MELD 실험용 인코더
    "HIDDEN_SIZE": 768,                 # RoBERTa-base hidden state 차원
    "PAD_TOKEN_ID": 1,                  # roberta-base의 <pad> 토큰 ID
    "FREEZE_ENCODER": False,             # 소규모 데이터셋에서는 encoder를 고정하고 상위 모듈만 학습
}

# ── Compressive Memory 설정 ────────────────────────────────
MEMORY_CONFIG = {
    "SLOT_DIM": 128,            # 메모리 슬롯 1개의 차원 (Projection 출력 차원)
    "FM_SIZE": 3,               # Fine Memory 크기 (r=3, 최근 3개 세그먼트 보존)
    "CM_SIZE": 4,               # Compressed Memory 크기 (k=4, 고정 슬롯)
    "CONV_KERNEL_SIZE": 2,      # 1D Conv 압축 함수의 커널 크기
    "USE_LEARNABLE_INIT": True, # 학습 가능한 초기 메모리 슬롯 사용 여부
}

# ── Attention 설정 ─────────────────────────────────────────
ATTENTION_CONFIG = {
    "NUM_HEADS": 8,             # Single-head Attention (Ablation에서 multi-head 비교)
    "DROPOUT": 0.1,             # Attention dropout 비율
}

# ── Classification Head 설정 ───────────────────────────────
HEAD_CONFIG = {
    "HIDDEN_DIM": 64,           # 중간 은닉층 차원 (128 → 64 → NUM_LABELS)
    "DROPOUT": 0.1,             # Head dropout 비율
}

# ── 학습 설정 ──────────────────────────────────────────────
TRAIN_CONFIG = {
    # 기본 학습 파라미터
    "BATCH_SIZE": 8,            # 실제 배치 크기 (유효 배치 = 8 × accum_steps(2) = 16)
    "EPOCHS": 20,               # 최대 에폭 수
    "SEED": 42,                 # 재현성을 위한 시드

    # 옵티마이저 (AdamW, 차등 학습률)
    "ENCODER_LR": 2e-5,         # RoBERTa fine-tuning 학습률
    "UPPER_LR": 5e-4,           # Memory, Attention, Head 학습률
    "WEIGHT_DECAY": 0.01,       # AdamW weight decay

    # Learning Rate Scheduler (Linear warmup + cosine decay)
    "WARMUP_RATIO": 0.06,        # 전체 step의 10% warmup
    "MIN_LR": 1e-6,             # cosine decay 최소 학습률

    # Early Stopping
    "PATIENCE": 7,              # validation loss 개선 없으면 조기 종료
    "MIN_DELTA": 1e-4,          # 개선으로 인정하는 최소 변화량

    # Loss 전략 ("L1": 마지막만, "L2": 동일 가중치, "L3": 시간 비례 가중치)
    "LOSS_STRATEGY": "L3",      # dialogue 최종 라벨 기준이므로 마지막 세그먼트 supervision이 가장 안전
    "POS_WEIGHT_CLIP": 10.0,    # pos_weight 최댓값 제한

    # Gradient 관리
    "MAX_GRAD_NORM": 1.0,       # gradient clipping 최댓값
    "CM_DETACH_INTERVAL": 0,    # 0이면 detach 안 함 (MELD는 세그먼트 수가 적으므로)
}

# ── 평가 설정 ──────────────────────────────────────────────
EVAL_CONFIG = {
    "THRESHOLD": 0.5,                   # multi-hot 이진화 기본 임계값
    "THRESHOLD_SEARCH_RANGE": (0.2, 0.8, 0.05),  # 클래스별 최적 threshold 탐색 범위
}

# ── Ablation 실험 설정 ─────────────────────────────────────
ABLATION_CONFIGS = {
    # A: Baseline (메모리 없음)
    "baseline": {
        "FM_SIZE": 0,
        "CM_SIZE": 0,
        "description": "메모리 없음, 마지막 세그먼트만 분류",
    },
    # B: Fine Memory만 사용
    "fm_only": {
        "FM_SIZE": 3,
        "CM_SIZE": 0,
        "description": "CM 제거, FM만 유지 (r=3)",
    },
    # C: 제안 모델 (Full) — 기본 설정과 동일
    "full": {
        "FM_SIZE": 3,
        "CM_SIZE": 4,
        "description": "FM(r=3) + CM(k=4) + 1D Conv (메인 실험)",
    },
    # D: 압축 함수 비교
    "mean_pooling": {
        "FM_SIZE": 3,
        "CM_SIZE": 4,
        "COMPRESS_FN": "mean",  # 1D Conv 대신 Mean Pooling
        "description": "압축 함수를 Mean Pooling으로 변경",
    },
    # E: k Ablation
    "cm_k1": {"CM_SIZE": 1, "description": "CM 슬롯 수 k=1"},
    "cm_k2": {"CM_SIZE": 2, "description": "CM 슬롯 수 k=2"},
    "cm_k8": {"CM_SIZE": 8, "description": "CM 슬롯 수 k=8"},
}
