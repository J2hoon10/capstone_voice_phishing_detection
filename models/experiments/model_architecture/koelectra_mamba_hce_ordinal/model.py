"""
koelectra_mamba_phishing6_hce / model.py

변경 사항 (vs koelectra_mamba_phishing5):
  1. Hierarchical Classification Head
     - superclass_head : 2-class (일반 / 피싱) 이진 분류
     - subclass_heads  : 그룹별 세부 분류
         * normal_head  : 3-class (일반 대화 3종)
         * phishing_head: 2-class (피싱 2종)
  2. Ordinal Auxiliary Head
     - aux_head : K-1 = 2 개의 독립 이진 로짓 (위험도 3단계 서수 회귀)
       → OrdinalRegressionLoss 와 함께 사용
  3. forward() 출력에 'super_logits', 'normal_logits', 'phishing_logits' 추가

backbone(KoELECTRA + Mamba)은 koelectra_mamba_phishing5 / streaming_belief_v5 에서
동적으로 임포트하여 재사용합니다.
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
    """
    Spec 2.2 / 3.2 의 계층적 분류를 담당하는 모듈.

    forward 입력 : (B, D) 또는 (B, T, D) 피처
    forward 출력 : dict
        super_logits   : (B, [T,] 2)         일반 / 피싱 이진 로짓
        normal_logits  : (B, [T,] 3)         일반 내 3종 세부 로짓
        phishing_logits: (B, [T,] 2)         피싱 내 2종 세부 로짓
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.superclass_head = nn.Linear(hidden_dim, NUM_SUPERCLASSES)   # 일반/피싱
        self.normal_head = nn.Linear(hidden_dim, 3)                       # 일반 3종
        self.phishing_head = nn.Linear(hidden_dim, 2)                     # 피싱 2종

    def forward(self, x: torch.Tensor) -> dict:
        x = self.proj(self.norm(x))
        return {
            "super_logits": self.superclass_head(x),
            "normal_logits": self.normal_head(x),
            "phishing_logits": self.phishing_head(x),
        }


# ── 주 모델 ─────────────────────────────────────────────────────────────────
class StreamingBeliefClassifier(BaseStreamingBeliefClassifier):
    """
    koelectra_mamba_phishing6_hce 모델.

    phishing5 대비 변경된 아키텍처:
      - head → HierarchicalHead (super / normal / phishing 로짓 분리 출력)
      - aux_head → K-1 = 2 로짓 (Ordinal Regression)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        d = self.d_model
        hidden = HEAD_CONFIG["HIDDEN_DIM"]
        drop = HEAD_CONFIG["DROPOUT"]

        # 부모 클래스의 head 를 계층적 헤드로 교체
        self.head = HierarchicalHead(d, hidden, drop)

        # Ordinal Regression 보조 헤드: K-1=2 이진 로짓
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

        # ── 세그먼트별 KoELECTRA 인코딩 ──────────────────────────────────
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

        # ── Mamba SSM ────────────────────────────────────────────────────
        x_stacked = torch.stack(segment_vectors, dim=1)   # (B, T, D)

        # 보조 헤드 입력: Mamba 통과 전 세그먼트 벡터
        aux_logits = self.aux_head(x_stacked)             # (B, T, 2) — ordinal K-1

        y = x_stacked
        for mamba, dropout in zip(self.mamba_layers, self.mamba_dropouts):
            y = dropout(mamba(y))                         # (B, T, D)

        # ── 계층적 분류 헤드 ─────────────────────────────────────────────
        # 시퀀스 전체에 대해 헤드 적용 후 마지막 유효 스텝 추출
        head_out_seq = self.head(y)                       # 각 (B, T, *)

        # _gather_last_logits 는 (list of (B, C)) 를 받아 (B, C) 반환
        # → 대신 직접 마지막 유효 인덱스로 gather
        last_idx = (num_segments - 1).clamp(min=0)        # (B,)

        def _gather(tensor: torch.Tensor) -> torch.Tensor:
            # tensor: (B, T, C)
            idx = last_idx.view(B, 1, 1).expand(B, 1, tensor.size(-1))
            return tensor.gather(1, idx).squeeze(1)       # (B, C)

        super_logits = _gather(head_out_seq["super_logits"])
        normal_logits = _gather(head_out_seq["normal_logits"])
        phishing_logits = _gather(head_out_seq["phishing_logits"])

        # ── 최종 5-class 로짓 재조합 ─────────────────────────────────────
        # 추론 시 argmax 로 사용할 통합 로짓을 구성:
        #   normal  0~2 → superclass 0 확률 * normal softmax
        #   phishing 3~4 → superclass 1 확률 * phishing softmax
        super_probs = torch.softmax(super_logits, dim=-1)         # (B, 2)
        normal_probs = torch.softmax(normal_logits, dim=-1)       # (B, 3)
        phishing_probs = torch.softmax(phishing_logits, dim=-1)   # (B, 2)

        logits_5 = torch.cat(
            [
                super_probs[:, 0:1] * normal_probs,       # 일반 3종
                super_probs[:, 1:2] * phishing_probs,     # 피싱 2종
            ],
            dim=-1,
        )  # (B, 5)  — 확률값이므로 학습 중 loss 계산은 개별 로짓 사용

        # per-step all_logits (evaluate 호환용 — 단순 CE 로 val loss 계산)
        all_logits = [
            torch.log(logits_5 + 1e-8)
        ] * T

        return {
            "logits": torch.log(logits_5 + 1e-8),        # (B, 5) log-prob
            "super_logits": super_logits,                  # (B, 2)
            "normal_logits": normal_logits,                # (B, 3)
            "phishing_logits": phishing_logits,            # (B, 2)
            "all_logits": all_logits,
            "aux_logits": aux_logits,                      # (B, T, 2) ordinal
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
