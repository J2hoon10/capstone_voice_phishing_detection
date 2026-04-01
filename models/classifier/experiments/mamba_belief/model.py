"""
BERT + Mamba SSM 스트리밍 분류 모델

전체 구조:
  세그먼트 c_t → RoBERTa Encoder → H^(t) ∈ R^(L×768)
  모든 세그먼트 concat → Mamba SSM (병렬 스캔) → 세그먼트별 split
  → mean pool → Classification Head → multi-hot 출력

핵심 설계:
  - BERT: 청크 내부 의미 파악 (양방향 attention, 청크 독립 처리)
  - Mamba SSM: 청크 간 Belief 누적 (선택적 상태 업데이트, O(L) 복잡도)
  - 학습: 전체 세그먼트 concat → Mamba parallel scan (효율적)
  - 추론: Mamba recurrent step mode 사용 가능 (O(1) 메모리)

mamba-ssm 패키지가 없는 환경(Windows)에서는 ssm_fallback.MambaPyTorch 를 사용한다.
"""

import io
from contextlib import redirect_stderr, redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    from mamba_ssm import Mamba
except ImportError:
    from ssm_fallback import MambaPyTorch as Mamba


IGNORED_LOAD_REPORT_PREFIXES = (
    "lm_head.",
    "pooler.",
    "roberta.embeddings.position_ids",
)


def load_encoder_backbone(encoder_name: str):
    """
    HuggingFace 모델 로드 출력을 캡처한 뒤,
    backbone fine-tuning에 의미 있는 항목만 출력한다.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        encoder = AutoModel.from_pretrained(encoder_name)

    noteworthy_rows = [
        line.rstrip()
        for line in buffer.getvalue().splitlines()
        if "|" in line
        and line.split("|", 1)[0].strip()
        and not line.split("|", 1)[0].strip().startswith(("-", "Key"))
        and not any(
            line.split("|", 1)[0].strip().startswith(p)
            for p in IGNORED_LOAD_REPORT_PREFIXES
        )
    ]

    if noteworthy_rows:
        print(f"[모델] {encoder_name} 로드 리포트 (backbone 관련 항목만 표시)")
        for row in noteworthy_rows:
            print(row)
    else:
        print(f"[모델] {encoder_name} 로드 완료")

    return encoder


class ClassificationHead(nn.Module):
    """분류 헤드: d_model → hidden_dim → num_labels (multi-hot)"""

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class BERTMambaClassifier(nn.Module):
    """
    RoBERTa + Mamba SSM 스트리밍 분류 모델.

    forward() 인터페이스는 compressive_memory/model.py 와 동일하여
    train.py / evaluate.py 의 코드를 최소 변경으로 재사용 가능하다.

    Args:
        encoder_name: HuggingFace 모델명 (예: "roberta-base")
        d_model: BERT hidden size (768)
        d_state: Mamba SSM state 차원
        d_conv: Mamba Conv1d 커널 크기
        expand: Mamba inner projection 확장 비율
        num_mamba_layers: Mamba 레이어 수 (1 또는 2)
        num_labels: 출력 레이블 수 (10)
        head_hidden_dim: Head 중간 차원
        dropout: Head dropout
        freeze_encoder: 인코더 파라미터 고정 여부
        skip_mamba: True이면 Mamba 없이 CLS 토큰만으로 분류 (baseline)
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_mamba_layers: int = 1,
        num_labels: int = 10,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        skip_mamba: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_labels = num_labels
        self.skip_mamba = skip_mamba

        # 1. RoBERTa 인코더
        self.encoder = load_encoder_backbone(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 2. Mamba SSM (skip_mamba=True이면 생략)
        if not skip_mamba:
            self.mamba_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(num_mamba_layers)
            ])
        else:
            self.mamba_layers = None

        # 3. Classification Head
        self.head = ClassificationHead(
            input_dim=d_model,
            hidden_dim=head_hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )

    def _encode_segment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        하나의 세그먼트를 BERT로 인코딩하여 전체 토큰 hidden state를 반환한다.

        Returns:
            (B, L, d_model) — CLS 토큰 포함 전체 시퀀스
        """
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        씬의 모든 세그먼트를 처리하여 분류 결과를 반환한다.

        Args:
            input_ids:      (B, max_S, L) 세그먼트별 토큰 ID
            attention_mask: (B, max_S, L) 세그먼트별 어텐션 마스크
            segment_mask:   (B, max_S)    실제 세그먼트=True, 패딩=False
            num_segments:   (B,)          각 씬의 실제 세그먼트 수

        Returns:
            dict:
                - logits:     (B, num_labels) 최종 분류 logits
                - all_logits: list[(B, num_labels)] 각 세그먼트별 logits
        """
        if self.skip_mamba:
            return self._forward_baseline(
                input_ids, attention_mask, segment_mask, num_segments
            )
        return self._forward_mamba(
            input_ids, attention_mask, segment_mask, num_segments
        )

    def _forward_mamba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
    ) -> dict:
        """
        BERT → concat → Mamba parallel scan → split → classify

        학습 전략:
          1. 각 세그먼트를 BERT로 독립 인코딩 → H^(t) ∈ (B, L, 768)
          2. 유효 세그먼트 토큰을 시간순으로 concat → H_all ∈ (B, T*L, 768)
          3. Mamba parallel scan → Y_all ∈ (B, T*L, 768)
             (Mamba가 토큰 경계를 넘어 청크 간 기억 누적)
          4. 세그먼트 단위로 split → mean pool → 분류
        """
        B, max_S, L = input_ids.shape
        device = input_ids.device
        enc_dtype = next(self.encoder.parameters()).dtype

        # ── Step 1: BERT 인코딩 (유효 세그먼트만) ──────────────
        all_H = []   # 유효 세그먼트의 BERT 출력 리스트
        for t in range(max_S):
            valid = segment_mask[:, t]  # (B,)
            if not valid.any():
                break

            H_t = torch.zeros(B, L, self.d_model, device=device, dtype=enc_dtype)
            H_t_valid = self._encode_segment(
                input_ids[:, t, :][valid],
                attention_mask[:, t, :][valid],
            )
            H_t[valid] = H_t_valid.to(enc_dtype)
            all_H.append(H_t)

        T = len(all_H)
        if T == 0:
            dummy = torch.zeros(B, self.num_labels, device=device)
            return {"logits": dummy, "all_logits": [dummy]}

        # ── Step 2: 전체 세그먼트 토큰 concat ──────────────────
        # H_all: (B, T*L, d_model)
        H_all = torch.cat(all_H, dim=1).float()

        # ── Step 3: Mamba parallel scan ────────────────────────
        # Y_all: (B, T*L, d_model)
        Y_all = H_all
        for mamba in self.mamba_layers:
            Y_all = mamba(Y_all)

        # ── Step 4: 세그먼트별 split → mean pool → classify ────
        all_logits = []
        for t in range(T):
            Y_t = Y_all[:, t * L:(t + 1) * L, :]   # (B, L, d_model)
            y_mean = Y_t.mean(dim=1)                 # (B, d_model)
            logits = self.head(y_mean)               # (B, num_labels)
            all_logits.append(logits)

        final_logits = self._gather_last_logits(all_logits, num_segments)
        return {"logits": final_logits, "all_logits": all_logits}

    def _forward_baseline(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_mask: torch.Tensor,
        num_segments: torch.Tensor,
    ) -> dict:
        """
        Baseline: Mamba 없이 각 세그먼트의 CLS 토큰으로 직접 분류.
        compressive_memory baseline과 동일한 역할.
        """
        B, max_S, L = input_ids.shape
        device = input_ids.device
        enc_dtype = next(self.encoder.parameters()).dtype

        all_logits = []
        for t in range(max_S):
            valid = segment_mask[:, t]
            if not valid.any():
                break

            cls_t = torch.zeros(B, self.d_model, device=device, dtype=enc_dtype)
            H_t_valid = self._encode_segment(
                input_ids[:, t, :][valid],
                attention_mask[:, t, :][valid],
            )
            # CLS 토큰(index 0)을 세그먼트 표현으로 사용
            cls_t[valid] = H_t_valid[:, 0, :].to(enc_dtype)

            logits = self.head(cls_t.float())
            all_logits.append(logits)

        if not all_logits:
            dummy = torch.zeros(B, self.num_labels, device=device)
            return {"logits": dummy, "all_logits": [dummy]}

        final_logits = self._gather_last_logits(all_logits, num_segments)
        return {"logits": final_logits, "all_logits": all_logits}

    def _gather_last_logits(
        self,
        all_logits: list[torch.Tensor],
        num_segments: torch.Tensor,
    ) -> torch.Tensor:
        """각 샘플의 마지막 유효 세그먼트 logits를 모은다."""
        B = num_segments.shape[0]
        device = all_logits[0].device
        final = torch.zeros(B, self.num_labels, device=device)

        for b in range(B):
            last_t = min(num_segments[b].item() - 1, len(all_logits) - 1)
            final[b] = all_logits[last_t][b]

        return final


def build_mamba_model(ablation_config: dict) -> BERTMambaClassifier:
    """
    config dict로부터 BERTMambaClassifier를 생성하는 팩토리 함수.

    Args:
        ablation_config: ABLATION_CONFIGS의 값 dict (오버라이드 설정)

    Returns:
        BERTMambaClassifier 인스턴스
    """
    from config import ENCODER_CONFIG, MAMBA_CONFIG, HEAD_CONFIG, MELD_CONFIG

    return BERTMambaClassifier(
        encoder_name=ENCODER_CONFIG["MODEL_NAME"],
        d_model=MAMBA_CONFIG["D_MODEL"],
        d_state=ablation_config.get("D_STATE", MAMBA_CONFIG["D_STATE"]),
        d_conv=MAMBA_CONFIG["D_CONV"],
        expand=MAMBA_CONFIG["EXPAND"],
        num_mamba_layers=ablation_config.get("NUM_LAYERS", MAMBA_CONFIG["NUM_LAYERS"]),
        num_labels=MELD_CONFIG["NUM_LABELS"],
        head_hidden_dim=HEAD_CONFIG["HIDDEN_DIM"],
        dropout=HEAD_CONFIG["DROPOUT"],
        freeze_encoder=ablation_config.get("FREEZE_ENCODER", ENCODER_CONFIG["FREEZE_ENCODER"]),
        skip_mamba=ablation_config.get("SKIP_MAMBA", False),
    )
