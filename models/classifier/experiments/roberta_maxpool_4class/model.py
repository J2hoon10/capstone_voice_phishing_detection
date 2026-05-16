"""
roberta_maxpool / model.py

roberta_avgpool 대비 변경 사항:
  - 세그먼트 풀링: 평균(AvgPool) → 최대(MaxPool)
    * 기존: (seg_repr * seg_mask).sum(dim=1) / seg_mask.sum(dim=1)
    * 변경: segment_mask로 패딩 마스킹 후 dim=1 방향 max
    * 동기: 피싱 탐지에서 가장 위험한 세그먼트의 특징을 보존하는 것이 중요할 수 있음
  - 토큰 풀링: MeanPooling 유지 (roberta_avgpool과 동일)
  - 인코더: KLUE-RoBERTa-base 유지
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from config import ENCODER_CONFIG, HEAD_CONFIG, NUM_LABELS


class MeanPooling(nn.Module):
    """패딩을 제외한 토큰 hidden states의 평균으로 세그먼트 표현 생성."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)   # (B, H)
        count = mask.sum(dim=1).clamp(min=1e-9)      # (B, 1)
        return summed / count                         # (B, H)


class RoBertaMaxPoolClassifier(nn.Module):
    """KLUE-RoBERTa + MeanPooling(토큰) + 세그먼트 Max Pooling + MLP.

    roberta_avgpool 대비 변경: 세그먼트 수준 풀링을 average → max로 교체.
    유효 세그먼트 중 각 특징 차원의 최대값을 사용하여 가장 두드러진 신호를 보존.
    """

    def __init__(
        self,
        encoder_name: str,
        num_labels: int,
        head_hidden_dim: int,
        head_dropout: float,
        freeze_embeddings: bool = True,
        freeze_lower_layers: int = 10,
        unfreeze_all: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pooling = MeanPooling()
        self.aux_head = nn.Linear(768, 3)
        self.head = nn.Sequential(
            nn.Linear(768, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_labels),
        )

        if not unfreeze_all:
            if freeze_embeddings:
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = False
            for i in range(freeze_lower_layers):
                for p in self.encoder.encoder.layer[i].parameters():
                    p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
        **kwargs,
    ) -> dict:
        B, S, _ = input_ids.shape
        enc_dtype = next(self.encoder.parameters()).dtype
        device = input_ids.device

        seg_repr = torch.zeros(B, S, 768, device=device, dtype=enc_dtype)

        for t in range(S):
            valid = segment_mask[:, t]
            if not valid.any():
                break
            out = self.encoder(
                input_ids=input_ids[:, t, :][valid],
                attention_mask=attention_mask[:, t, :][valid],
            )
            x_valid = self.pooling(out.last_hidden_state, attention_mask[:, t, :][valid])
            seg_repr[valid, t] = x_valid

        seg_repr = seg_repr.float()

        aux_logits = self.aux_head(seg_repr)  # (B, S, 3)

        # Max pooling: 패딩 세그먼트를 -inf로 마스킹한 뒤 dim=1 방향 max
        seg_repr_masked = seg_repr.masked_fill(
            ~segment_mask.unsqueeze(-1).expand_as(seg_repr), float("-inf")
        )
        pooled = seg_repr_masked.max(dim=1).values  # (B, H)

        logits = self.head(pooled)  # (B, num_labels)
        return {"logits": logits, "aux_logits": aux_logits}


def build_model() -> RoBertaMaxPoolClassifier:
    return RoBertaMaxPoolClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        num_labels=NUM_LABELS,
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
        freeze_embeddings=True,
        freeze_lower_layers=10,
        unfreeze_all=False,
    )
