"""
roberta_mamba_freeze_init / model.py

  - 인코더: KLUE-RoBERTa-base (klue/roberta-base)
  - NUM_LAYERS=1, WINDOW_SIZE=128, STRIDE=100 (표준 window)
  - build_model(): unfreeze_all=True
    encoder freeze 전략은 train.py 에서 직접 관리.
    (처음 FREEZE_INIT_EPOCHS=3 동안 전체 freeze → 이후 layer[10~11] unfreeze)

  - forward() 반환값에 "anisotropy_metric" 포함
    Mamba 입력 직전 유효 세그먼트 벡터들의 평균 코사인 유사도 (비등방성 진단용)
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from transformers import AutoModel

from config import (
    ENCODER_CONFIG,
    HEAD_CONFIG,
    MAMBA_CONFIG,
    NUM_LABELS,
    NUM_SUPERCLASSES,
)

_IGNORED_LOAD_PREFIXES = (
    "lm_head.", "pooler.", "roberta.embeddings.position_ids",
    "electra.embeddings.position_ids", "discriminator_predictions.",
    "generator_predictions.", "generator_lm_head.",
)


def _load_encoder_backbone(encoder_name: str) -> nn.Module:
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)
    rows = [
        line.rstrip() for line in buf.getvalue().splitlines()
        if "|" in line
        and (key := line.split("|", 1)[0].strip())
        and key not in ("Key",)
        and not key.startswith("-")
        and not any(key.startswith(p) for p in _IGNORED_LOAD_PREFIXES)
    ]
    if rows:
        print(f"[model] {encoder_name} load report (backbone only)")
        for row in rows:
            print(row)
    else:
        print(f"[model] {encoder_name} loaded")
    return encoder


class AttentionWeightedPooling(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.W_b = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        H_tokens = hidden_states[:, 1:-1, :]
        mask_tokens = attention_mask[:, 1:-1]
        e = self.W_a(torch.tanh(self.W_b(H_tokens)))
        e = e.masked_fill(mask_tokens.unsqueeze(-1) == 0, -1e4)
        alpha = torch.softmax(e, dim=1)
        alpha = alpha * mask_tokens.unsqueeze(-1).float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (alpha * H_tokens).sum(dim=1)


class HierarchicalHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.superclass_head = nn.Linear(hidden_dim, NUM_SUPERCLASSES)
        self.normal_head = nn.Linear(hidden_dim, 2)
        self.phishing_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.proj(self.norm(x))
        return {
            "super_logits": self.superclass_head(x),
            "normal_logits": self.normal_head(x),
            "phishing_logits": self.phishing_head(x),
        }


class StreamingBeliefClassifier(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_mamba_layers: int,
        mamba_dropout: float,
        num_labels: int,
        head_hidden_dim: int,
        head_dropout: float,
        freeze_embeddings: bool = False,
        freeze_lower_layers: int = 0,
        unfreeze_all: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        self.encoder = _load_encoder_backbone(encoder_name)

        if not unfreeze_all:
            if freeze_embeddings and hasattr(self.encoder, "embeddings"):
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = False
            layers = None
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                layers = self.encoder.encoder.layer
            if layers and freeze_lower_layers > 0:
                for i in range(min(int(freeze_lower_layers), len(layers))):
                    for p in layers[i].parameters():
                        p.requires_grad = False

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
                lora_cfg = LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    bias="none", task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=lora_target_modules or ["query", "key", "value"],
                )
                self.encoder = get_peft_model(self.encoder, lora_cfg)
            except ImportError:
                print("[warn] peft not installed. continue without LoRA.")

        self.pooling = AttentionWeightedPooling(hidden_dim=d_model)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_mamba_layers)
        ])
        self.mamba_dropouts = nn.ModuleList([nn.Dropout(mamba_dropout) for _ in range(num_mamba_layers)])
        self.head = HierarchicalHead(d_model, head_hidden_dim, head_dropout)
        self.aux_head = nn.Linear(d_model, 2)

    def _encode_segment(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
        **kwargs,
    ) -> dict:
        B, max_S, _ = input_ids.shape
        device = input_ids.device
        enc_dtype = next(self.encoder.parameters()).dtype

        segment_vectors = []
        for t in range(max_S):
            valid = segment_mask[:, t]
            if not valid.any():
                break
            x_t = torch.zeros(B, self.d_model, device=device, dtype=enc_dtype)
            H_valid = self._encode_segment(
                input_ids=input_ids[:, t, :][valid],
                attention_mask=attention_mask[:, t, :][valid],
            )
            x_valid = self.pooling(H_valid, attention_mask[:, t, :][valid])
            x_t[valid] = x_valid.to(enc_dtype)
            segment_vectors.append(x_t.float())

        T = len(segment_vectors)
        if T == 0:
            dummy_logits = torch.zeros(B, NUM_LABELS, device=device)
            return {
                "logits": dummy_logits,
                "super_logits": torch.zeros(B, 2, device=device),
                "normal_logits": torch.zeros(B, 2, device=device),
                "phishing_logits": torch.zeros(B, 2, device=device),
                "all_logits": [dummy_logits],
                "aux_logits": torch.zeros(B, max_S, 2, device=device),
                "anisotropy_metric": torch.tensor(0.0, device=device),
            }

        x_stacked = torch.stack(segment_vectors, dim=1)   # (B, T, D)
        aux_logits = self.aux_head(x_stacked)             # (B, T, 2)

        with torch.no_grad():
            valid_flat = segment_mask[:, :T].reshape(-1)
            vecs_flat  = x_stacked.detach().reshape(-1, x_stacked.size(-1))
            valid_vecs = vecs_flat[valid_flat]
            N = valid_vecs.shape[0]
            if N > 1:
                normed        = F.normalize(valid_vecs, p=2, dim=-1)
                sim_mat       = normed @ normed.T
                off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=sim_mat.device)
                avg_similarity = sim_mat[off_diag_mask].mean()
            else:
                avg_similarity = x_stacked.new_tensor(0.0)

        y = x_stacked
        for mamba, dropout in zip(self.mamba_layers, self.mamba_dropouts):
            y = dropout(mamba(y))

        head_out_seq = self.head(y)
        last_idx = (num_segments - 1).clamp(min=0)

        def _gather(tensor: torch.Tensor) -> torch.Tensor:
            idx = last_idx.view(B, 1, 1).expand(B, 1, tensor.size(-1))
            return tensor.gather(1, idx).squeeze(1)

        super_logits    = _gather(head_out_seq["super_logits"])
        normal_logits   = _gather(head_out_seq["normal_logits"])
        phishing_logits = _gather(head_out_seq["phishing_logits"])

        super_probs    = torch.softmax(super_logits, dim=-1)
        normal_probs   = torch.softmax(normal_logits, dim=-1)
        phishing_probs = torch.softmax(phishing_logits, dim=-1)

        logits_5 = torch.cat(
            [
                super_probs[:, 0:1] * normal_probs,
                super_probs[:, 1:2] * phishing_probs,
            ],
            dim=-1,
        )

        all_logits = [torch.log(logits_5 + 1e-8)] * T

        return {
            "logits": torch.log(logits_5 + 1e-8),
            "super_logits": super_logits,
            "normal_logits": normal_logits,
            "phishing_logits": phishing_logits,
            "all_logits": all_logits,
            "aux_logits": aux_logits,
            "anisotropy_metric": avg_similarity,
        }


def build_model() -> StreamingBeliefClassifier:
    return StreamingBeliefClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        d_model=MAMBA_CONFIG["D_MODEL"],
        d_state=MAMBA_CONFIG["D_STATE"],
        d_conv=MAMBA_CONFIG["D_CONV"],
        expand=MAMBA_CONFIG["EXPAND"],
        num_mamba_layers=MAMBA_CONFIG["NUM_LAYERS"],
        mamba_dropout=MAMBA_CONFIG["DROPOUT"],
        num_labels=NUM_LABELS,
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
        freeze_embeddings=False,
        freeze_lower_layers=0,
        unfreeze_all=True,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["query", "key", "value"],
    )
