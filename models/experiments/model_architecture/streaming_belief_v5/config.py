"""
Streaming Belief v5 experiment configuration.

Phase 1 policy:
- 20-class dialogue-level dataset (same preprocessing schema as compressive_memory_kr).
- Keep a dedicated local cache in this experiment's own data directory.
"""

from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # allows data copy/prep utilities to run without torch
    torch = None


# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
DATA_ROOT = PROJECT_ROOT / "normal_data_reference"
COMPRESSIVE_MEMORY_DATA_DIR = (
    PROJECT_ROOT
    / "models"
    / "classifier"
    / "experiments"
    / "compressive_memory_kr"
    / "data"
)
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"


# Device / GPU
if torch is not None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

GPU_CONFIG = {
    "ENABLED": True,  # mixed precision/autocast on/off
    "DTYPE": "bf16",  # "bf16" or "fp16"
    "TF32_MATMUL": True,  # faster matmul on Ampere+
    "TF32_CUDNN": True,  # faster cudnn kernels on Ampere+
    "TARGET_VRAM_GB": 14.0,  # logging/monitoring target only
    "GRADIENT_ACCUMULATION_STEPS": 2,  # effective_batch = BATCH_SIZE * this
    "EMPTY_CACHE_INTERVAL": 0,  # set >0 to call torch.cuda.empty_cache periodically
    "CUDNN_BENCHMARK": True,  # optimize cudnn kernel selection
    "PIN_MEMORY": True,  # dataloader pin_memory
    "NUM_WORKERS": 0,  # dataloader workers
}


# Same label set as compressive_memory_kr (Phase 1, 20-class)
DATA_CONFIG = {
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
    "MAX_LENGTH": 128,  # kept for compatibility; window size is controlled by WINDOW_CONFIG
    "UTTERANCE_FORMAT": "{speaker_id}: {norm_text}",  # text template per utterance
    "SPLIT_RATIOS": {"train": 0.9, "val": 0.1},  # train/val split from normal_data_reference/Training
    "SPLIT_SEED": 42,  # deterministic split seed
    "SUBJECT_NORMALIZATION": {
        "상거래전반": "상거래 전반",
    },
}


# Overlap sliding-window configuration (token-level)
WINDOW_CONFIG = {
    "WINDOW_SIZE": 128,  # includes [CLS], [SEP]
    "STRIDE": 100,  # overlap = WINDOW_SIZE - STRIDE
}


# Encoder
ENCODER_CONFIG = {
    "MODEL_NAME": "monologg/koelectra-base-v3-discriminator",
    "HIDDEN_SIZE": 768,
    "PAD_TOKEN_ID": 0,
    "FREEZE_EMBEDDINGS": True,  # freeze token embeddings
    "FREEZE_LOWER_LAYERS": 10,  # freeze lower transformer blocks (1..N)
    "UNFREEZE_ALL": False,  # if True, ignore freeze options
    "USE_LORA": False,  # enable LoRA adapter training
    "LORA_R": 8,  # LoRA rank
    "LORA_ALPHA": 16,  # LoRA alpha scaling
    "LORA_DROPOUT": 0.1,  # LoRA dropout
    "LORA_TARGET_MODULES": ["query", "key", "value"],  # LoRA target submodules
}


# Pooling
POOLING_CONFIG = {
    "HIDDEN_SIZE": 768,
    "EXCLUDE_SPECIAL_TOKENS": True,
}


# Mamba
MAMBA_CONFIG = {
    "D_MODEL": 768,
    "D_STATE": 16,  # SSM state size (try 16/32/64)
    "D_CONV": 4,  # local conv kernel size in Mamba block
    "EXPAND": 2,  # expansion factor in Mamba inner projection
    "NUM_LAYERS": 2,  # stack depth
    "DROPOUT": 0.1,  # dropout after each Mamba layer
}


# Classification head
HEAD_CONFIG = {
    "HIDDEN_DIM": 64,  # MLP hidden dimension
    "DROPOUT": 0.1,  # MLP dropout
}


# Training
TRAIN_CONFIG = {
    "BATCH_SIZE": 8,  # per-step mini-batch size
    "EPOCHS": 10,  # total training epochs
    "SEED": 42,
    "ENCODER_LR": 2e-5,  # lower LR for pretrained encoder
    "UPPER_LR": 5e-4,  # higher LR for pooling/mamba/head
    "WEIGHT_DECAY": 0.01,  # AdamW weight decay
    "WARMUP_RATIO": 0.06,  # linear warmup ratio before cosine decay
    "MIN_LR": 1e-6,  # floor LR in cosine schedule
    "PATIENCE": 5,  # early stopping patience (macro F1)
    "MIN_DELTA": 1e-4,  # min F1 improvement to reset patience
    "FOCAL_GAMMA": 2.0,  # focal loss gamma
    # Phase 1 default: uniform temporal weighting (plan v6)
    "LAMBDA_ALPHA": 0.0,  # lambda_t = exp(-alpha * t)
    "MAX_GRAD_NORM": 1.0,  # gradient clipping norm
}


# Evaluation
EVAL_CONFIG = {
    "TOP_K": 1,
    "ENTROPY_THRESHOLD": 0.3,  # normalized entropy threshold
}


# Ablation settings
ABLATION_CONFIGS = {
    "v5_default": {
        "description": "KoELECTRA freeze + attention pooling + 2-layer mamba (d_state=16)",
    },
    "v5_mamba1": {
        "NUM_LAYERS": 1,
        "description": "Mamba layer=1 ablation",
    },
    "v5_dstate32": {
        "D_STATE": 32,
        "description": "Mamba d_state=32 ablation",
    },
    "v5_dstate64": {
        "D_STATE": 64,
        "description": "Mamba d_state=64 ablation",
    },
    "v5_lora_r8": {
        "UNFREEZE_ALL": True,
        "USE_LORA": True,
        "FREEZE_EMBEDDINGS": False,
        "FREEZE_LOWER_LAYERS": 0,
        "description": "LoRA(rank=8, alpha=16) + Mamba2",
    },
    "v5_unfreeze_all": {
        "UNFREEZE_ALL": True,
        "FREEZE_EMBEDDINGS": False,
        "FREEZE_LOWER_LAYERS": 0,
        "description": "Full encoder fine-tuning + Mamba2",
    },
    "v5_lambda005": {
        "LAMBDA_ALPHA": 0.05,
        "description": "Temporal focal weight alpha=0.05",
    },
    "v5_lambda01": {
        "LAMBDA_ALPHA": 0.1,
        "description": "Temporal focal weight alpha=0.1",
    },
    "v5_lambda02": {
        "LAMBDA_ALPHA": 0.2,
        "description": "Temporal focal weight alpha=0.2",
    },
}


# Main tuning cheat sheet for quick experiment control.
HYPERPARAMETER_GUIDE = {
    "data_resolution": {
        "WINDOW_CONFIG.WINDOW_SIZE": "Token length of each overlap window (including special tokens).",
        "WINDOW_CONFIG.STRIDE": "Window shift; smaller stride means larger overlap and more chunks.",
        "DATA_CONFIG.MAX_LENGTH": "Compatibility field; actual chunking uses WINDOW_CONFIG.",
        "DATA_CONFIG.SPLIT_SEED": "Controls train/val split reproducibility.",
    },
    "compute_and_batching": {
        "TRAIN_CONFIG.BATCH_SIZE": "Main VRAM/throughput knob. Increase until stable, then tune accumulation.",
        "GPU_CONFIG.GRADIENT_ACCUMULATION_STEPS": "Simulates larger batch size without increasing instantaneous VRAM.",
        "GPU_CONFIG.DTYPE": "bf16 is preferred on modern GPUs; fp16 may need loss scaling.",
    },
    "optimizer_and_schedule": {
        "TRAIN_CONFIG.ENCODER_LR": "Lower LR for pretrained KoELECTRA weights.",
        "TRAIN_CONFIG.UPPER_LR": "Higher LR for newly initialized pooling/Mamba/head layers.",
        "TRAIN_CONFIG.WEIGHT_DECAY": "Regularization strength (AdamW).",
        "TRAIN_CONFIG.WARMUP_RATIO": "Warmup portion before cosine decay.",
        "TRAIN_CONFIG.MIN_LR": "Lower bound of scheduler decay.",
    },
    "sequence_modeling": {
        "MAMBA_CONFIG.D_STATE": "Primary capacity knob for temporal belief memory (16/32/64).",
        "MAMBA_CONFIG.NUM_LAYERS": "Temporal depth (1 or 2 in ablations).",
        "MAMBA_CONFIG.DROPOUT": "Regularization for Mamba outputs.",
    },
    "loss_and_early_exit": {
        "TRAIN_CONFIG.FOCAL_GAMMA": "Higher gamma emphasizes hard examples.",
        "TRAIN_CONFIG.LAMBDA_ALPHA": "Temporal weighting; 0.0 is uniform, higher values emphasize earlier segments.",
        "TRAIN_CONFIG.PATIENCE": "How long to wait for validation macro-F1 improvement.",
        "TRAIN_CONFIG.MIN_DELTA": "Minimum F1 gain counted as improvement.",
    },
    "encoder_finetuning": {
        "ENCODER_CONFIG.FREEZE_LOWER_LAYERS": "Freeze lower transformer blocks for stability/speed.",
        "ENCODER_CONFIG.UNFREEZE_ALL": "Full fine-tuning switch.",
        "ENCODER_CONFIG.USE_LORA": "Parameter-efficient adaptation toggle.",
        "ENCODER_CONFIG.LORA_R/LORA_ALPHA/LORA_DROPOUT": "LoRA capacity and regularization.",
    },
}
