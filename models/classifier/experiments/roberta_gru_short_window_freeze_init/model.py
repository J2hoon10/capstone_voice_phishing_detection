"""
roberta_gru_short_window_freeze_init / model.py

roberta_mamba_short_window_freeze_init 과 동일한 구조에서
Mamba SSM을 2-layer 단방향 GRU로 교체.
  - 인코더: KLUE-RoBERTa-base (klue/roberta-base)
  - WINDOW_SIZE=64, STRIDE=32
  - AttentionWeightedPooling (CLS/SEP 제외, W_b+W_a 2-layer attention)
  - GRU(input=768, hidden=768, num_layers=2, dropout=0.1) — 단방향
  - HierarchicalHead: 일반/피싱 superclass → 세부 클래스 계층적 분류
  - aux_head: 세그먼트별 이진 위험도 (GRU 입력 전 raw 벡터에서 예측)
  - encoder freeze 전략은 train.py에서 관리 (freeze_init)
"""

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
from transformers import AutoModel

from config import (
    ENCODER_CONFIG,
    HEAD_CONFIG,
    NUM_LABELS,
    NUM_SUPERCLASSES,
    RNN_CONFIG,
)


def load_encoder_backbone(encoder_name: str) -> nn.Module:
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        encoder = AutoModel.from_pretrained(encoder_name)
    print(f"[model] {encoder_name} loaded")
    return encoder


class AttentionWeightedPooling(nn.Module):
    """CLS(index 0)와 SEP(마지막) 제외, 실제 내용 토큰에 대해 2-layer attention 가중합."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.W_b = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        H_tokens = hidden_states[:, 1:-1, :]      # (B, L-2, D) — CLS/SEP 제외
        mask_tokens = attention_mask[:, 1:-1]      # (B, L-2)

        e = self.W_a(torch.tanh(self.W_b(H_tokens)))          # (B, L-2, 1)
        e = e.masked_fill(mask_tokens.unsqueeze(-1) == 0, -1e4)
        alpha = torch.softmax(e, dim=1)
        alpha = alpha * mask_tokens.unsqueeze(-1).float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return (alpha * H_tokens).sum(dim=1)                   # (B, D)


class HierarchicalHead(nn.Module):
    """일반(3) / 피싱(2) superclass → subclass 계층적 분류 헤드."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.superclass_head = nn.Linear(hidden_dim, NUM_SUPERCLASSES)
        self.normal_head = nn.Linear(hidden_dim, 3)
        self.phishing_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.proj(self.norm(x))
        return {
            "super_logits": self.superclass_head(x),
            "normal_logits": self.normal_head(x),
            "phishing_logits": self.phishing_head(x),
        }


class StreamingBeliefClassifier(nn.Module):
    """RoBERTa + AttentionWeightedPooling + 2-layer GRU + HierarchicalHead."""

    def __init__(
        self,
        encoder_name: str,
        d_model: int,
        rnn_hidden: int,
        rnn_layers: int,
        rnn_dropout: float,
        num_labels: int,
        head_hidden_dim: int,
        head_dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels

        self.encoder = load_encoder_backbone(encoder_name)
        self.pooling = AttentionWeightedPooling(hidden_dim=d_model)

        # inter-layer dropout은 num_layers > 1일 때만 유효
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )

        self.head = HierarchicalHead(rnn_hidden, head_hidden_dim, head_dropout)
        self.aux_head = nn.Linear(d_model, 2)  # GRU 이전 raw 세그먼트 벡터에서 예측

    def _encode_segment(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, max_S, L)
        attention_mask: torch.Tensor,  # (B, max_S, L)
        segment_mask: torch.Tensor,    # (B, max_S)
        num_segments: torch.Tensor,    # (B,)
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
                input_ids[:, t, :][valid],
                attention_mask[:, t, :][valid],
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
                "normal_logits": torch.zeros(B, 3, device=device),
                "phishing_logits": torch.zeros(B, 2, device=device),
                "all_logits": [dummy_logits],
                "aux_logits": torch.zeros(B, max_S, 2, device=device),
            }

        x_stacked = torch.stack(segment_vectors, dim=1)   # (B, T, D)
        aux_logits = self.aux_head(x_stacked)             # (B, T, 2) — GRU 전 raw 벡터

        rnn_out, _ = self.rnn(x_stacked)                  # (B, T, rnn_hidden)

        head_out_seq = self.head(rnn_out)
        last_idx = (num_segments - 1).clamp(min=0)

        def _gather(tensor: torch.Tensor) -> torch.Tensor:
            idx = last_idx.view(B, 1, 1).expand(B, 1, tensor.size(-1))
            return tensor.gather(1, idx).squeeze(1)

        super_logits = _gather(head_out_seq["super_logits"])
        normal_logits = _gather(head_out_seq["normal_logits"])
        phishing_logits = _gather(head_out_seq["phishing_logits"])

        super_probs = torch.softmax(super_logits, dim=-1)
        normal_probs = torch.softmax(normal_logits, dim=-1)
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
        }


def build_model() -> StreamingBeliefClassifier:
    return StreamingBeliefClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        d_model=RNN_CONFIG["D_MODEL"],
        rnn_hidden=RNN_CONFIG["HIDDEN_SIZE"],
        rnn_layers=RNN_CONFIG["NUM_LAYERS"],
        rnn_dropout=RNN_CONFIG["DROPOUT"],
        num_labels=NUM_LABELS,
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
    )
