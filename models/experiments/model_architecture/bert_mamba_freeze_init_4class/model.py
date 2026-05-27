"""
roberta_mamba_freeze_init / model.py

roberta_mamba_short_window_freeze_init 과 동일한 아키텍처.
  - 인코더: KLUE-RoBERTa-base (klue/roberta-base)
  - NUM_LAYERS=1, WINDOW_SIZE=128, STRIDE=100 (표준 window)
  - build_model(): unfreeze_all=True
    encoder freeze 전략은 train.py 에서 직접 관리.
    (처음 FREEZE_INIT_EPOCHS=3 동안 전체 freeze → 이후 layer[10~11] unfreeze)

추가:
  - forward() 반환값에 "anisotropy_metric" 포함
    Mamba 입력 직전 유효 세그먼트 벡터들의 평균 코사인 유사도 (비등방성 진단용)
"""

from pathlib import Path
import sys
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # ── 비등방성(Anisotropy) 지표: Mamba 입력 직전 ──────────────────────
        # 패딩 더미를 제외한 유효 세그먼트 벡터 간 평균 코사인 유사도
        # 값이 1에 가까울수록 벡터들이 한 방향으로 밀집(anisotropic)됨을 의미
        with torch.no_grad():
            valid_flat = segment_mask[:, :T].reshape(-1)                    # (B*T,) bool
            vecs_flat  = x_stacked.detach().reshape(-1, x_stacked.size(-1)) # (B*T, D)
            valid_vecs = vecs_flat[valid_flat]                              # (N, D)
            N = valid_vecs.shape[0]
            if N > 1:
                normed        = F.normalize(valid_vecs, p=2, dim=-1)        # (N, D)
                sim_mat       = normed @ normed.T                            # (N, N)
                off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=sim_mat.device)
                avg_similarity = sim_mat[off_diag_mask].mean()
            else:
                avg_similarity = x_stacked.new_tensor(0.0)
        # ────────────────────────────────────────────────────────────────────

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
    # unfreeze_all=True: 모든 파라미터 trainable로 시작
    # encoder의 실제 freeze/unfreeze는 train.py의 freeze_init 전략으로 관리
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
