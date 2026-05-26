"""
Model wrapper for klue/bert-base sequence classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, AutoTokenizer

from config import DATA_CONFIG, ENCODER_CONFIG


def build_model() -> BertForSequenceClassification:
    """Load klue/bert-base with a classification head."""
    model = BertForSequenceClassification.from_pretrained(
        ENCODER_CONFIG["MODEL_NAME"],
        num_labels=DATA_CONFIG["NUM_LABELS"],
    )
    return model


def load_model_from_checkpoint(checkpoint_path: str) -> BertForSequenceClassification:
    """Load a fine-tuned model from a checkpoint."""
    model = build_model()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    return model


def get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(ENCODER_CONFIG["MODEL_NAME"])


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_linear_modules(model: nn.Module, skip_patterns: list[str] | None = None) -> dict[str, nn.Linear]:
    """Get all nn.Linear modules, optionally filtering by skip patterns."""
    skip = skip_patterns or []
    linear_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(pat in name for pat in skip):
                continue
            linear_modules[name] = module
    return linear_modules


def model_size_bytes(model: nn.Module) -> int:
    """Calculate model size in bytes (FP32)."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes
