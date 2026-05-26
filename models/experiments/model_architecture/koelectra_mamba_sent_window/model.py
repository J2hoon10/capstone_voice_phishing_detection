"""
koelectra_mamba_sent_window / model.py

hce_ordinal 과 동일한 아키텍처 사용.
전처리 변경(문장 단위 의미 보존 분할)으로 세그먼트 수 증가 및
피싱 키워드가 세그먼트 경계에서 잘리는 현상 방지.
"""

from pathlib import Path
import sys
import importlib.util
import torch
import torch.nn as nn

from config import (
    ENCODER_CONFIG,
    HEAD_CONFIG,
    MAMBA_CONFIG,
    NUM_LABELS,
    NUM_SUPERCLASSES,
)

# ── streaming_belief_v5 base 모델 동적 로드 ────────────────────────────────
SRC_DIR = Path(__file__).resolve().parents[1] / "streaming_belief_v5"
model_path = SRC_DIR / "model.py"
if not model_path.exists():
    raise FileNotFoundError(f"Cannot find streaming belief model file: {model_path}")

spec = importlib.util.spec_from_file_location("streaming_belief_v5_model", model_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {model_path}")
streaming_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = streaming_module
spec.loader.exec_module(streaming_module)
BaseStreamingBeliefClassifier = streaming_module.StreamingBeliefClassifier


# ── 계층적 분류 헤드 ────────────────────────────────────────────────────────
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
        self.normal_head = nn.Linear(hidden_dim, 3)
        self.phishing_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.proj(self.norm(x))
        return {
            "super_logits": self.superclass_head(x),
            "normal_logits": self.normal_head(x),
            "phishing_logits": self.phishing_head(x),
        }


# ── 주 모델 ─────────────────────────────────────────────────────────────────
class StreamingBeliefClassifier(BaseStreamingBeliefClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d = self.d_model
        hidden = HEAD_CONFIG["HIDDEN_DIM"]
        drop = HEAD_CONFIG["DROPOUT"]

        self.head = HierarchicalHead(d, hidden, drop)
        self.aux_head = nn.Linear(d, 2)

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
            dummy_super = torch.zeros(B, 2, device=device)
            dummy_normal = torch.zeros(B, 3, device=device)
            dummy_phishing = torch.zeros(B, 2, device=device)
            dummy_logits = torch.zeros(B, NUM_LABELS, device=device)
            dummy_aux = torch.zeros(B, max_S, 2, device=device)
            return {
                "logits": dummy_logits,
                "super_logits": dummy_super,
                "normal_logits": dummy_normal,
                "phishing_logits": dummy_phishing,
                "all_logits": [dummy_logits],
                "aux_logits": dummy_aux,
            }

        x_stacked = torch.stack(segment_vectors, dim=1)   # (B, T, D)
        aux_logits = self.aux_head(x_stacked)             # (B, T, 2)

        y = x_stacked
        for mamba, dropout in zip(self.mamba_layers, self.mamba_dropouts):
            y = dropout(mamba(y))

        head_out_seq = self.head(y)
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
        d_model=MAMBA_CONFIG["D_MODEL"],
        d_state=MAMBA_CONFIG["D_STATE"],
        d_conv=MAMBA_CONFIG["D_CONV"],
        expand=MAMBA_CONFIG["EXPAND"],
        num_mamba_layers=MAMBA_CONFIG["NUM_LAYERS"],
        mamba_dropout=MAMBA_CONFIG["DROPOUT"],
        num_labels=NUM_LABELS,
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        head_dropout=HEAD_CONFIG["DROPOUT"],
        freeze_embeddings=True,
        freeze_lower_layers=10,
        unfreeze_all=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["query", "key", "value"],
    )
