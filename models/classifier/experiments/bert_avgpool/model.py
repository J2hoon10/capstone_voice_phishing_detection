import torch
import torch.nn as nn
from transformers import AutoModel

from config import ENCODER_CONFIG, HEAD_CONFIG, NUM_LABELS


class MeanPooling(nn.Module):
    """패딩을 제외한 토큰 hidden states의 평균으로 세그먼트 표현 생성."""

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, L, H), attention_mask: (B, L)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (hidden_states * mask).sum(dim=1)   # (B, H)
        count = mask.sum(dim=1).clamp(min=1e-9)      # (B, 1)
        return summed / count                         # (B, H)


class BertAvgPoolClassifier(nn.Module):
    """KLUE-BERT + MeanPooling(토큰) + 세그먼트 평균 풀링 + MLP.

    koelectra_phishing5 대비 변경 사항:
      - 인코더: KoELECTRA-base-v3 → KLUE-BERT-base (klue/bert-base)
      - 토큰 풀링: AttentionWeightedPooling → MeanPooling
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

        seg_mask = segment_mask.unsqueeze(-1).float()  # (B, S, 1)
        pooled = (seg_repr * seg_mask).sum(dim=1) / seg_mask.sum(dim=1).clamp(min=1)  # (B, H)

        logits = self.head(pooled)  # (B, num_labels)
        return {"logits": logits, "aux_logits": aux_logits}


def build_model() -> BertAvgPoolClassifier:
    return BertAvgPoolClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        num_labels=NUM_LABELS,
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
        freeze_embeddings=True,
        freeze_lower_layers=10,
        unfreeze_all=False,
    )
