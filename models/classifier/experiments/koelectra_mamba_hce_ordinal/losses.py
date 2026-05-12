"""
koelectra_mamba_phishing6_hce / losses.py

Spec §3 의 두 가지 손실 함수를 구현합니다.

1. OrdinalRegressionLoss  (Spec §3.1)
   위험도 3단계(안전/의심/위험)의 순서 관계를 학습하기 위해
   K-1 = 2 개의 독립 이진 분류 문제로 변환합니다.

   Label y ∈ {0, 1, 2} → 이진 타겟 벡터 b ∈ {0,1}^(K-1)
     b_k = 1  if y > k   (k=0,1)
   수식:
     L_aux = -Σ_k [ b_k * log σ(f_k) + (1-b_k) * log(1-σ(f_k)) ]

2. HierarchicalCrossEntropyLoss  (Spec §3.2)
   5-class 를 계층(일반 vs 피싱) 구조로 분해합니다.
     L_main = L_super + λ * Σ L_sub
   Focal Loss 가중치를 선택적으로 적용할 수 있습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOSS_CONFIG, SUPERCLASS_MAP


# ────────────────────────────────────────────────────────────────────────────
# 1.  Ordinal Regression Loss
# ────────────────────────────────────────────────────────────────────────────
class OrdinalRegressionLoss(nn.Module):
    """
    K = 3 클래스 서수 회귀 손실.
    모델 출력: (B, T, K-1) 이진 로짓
    레이블:    (B, T)  long ∈ {0,1,2},  패딩은 ignore_index (-100)
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, T, K-1)
        targets: (B, T)
        """
        B, T, Km1 = logits.shape
        K = Km1 + 1

        # 유효 마스크
        valid = targets != self.ignore_index              # (B, T)

        if not valid.any():
            return logits.sum() * 0.0

        # 이진 타겟 구성: b_k = [y > k]  for k in 0 … K-2
        # → (B, T, K-1)
        y_exp = targets.unsqueeze(-1).expand(B, T, Km1)  # (B, T, K-1)
        thresholds = torch.arange(Km1, device=logits.device).view(1, 1, Km1)
        binary_targets = (y_exp > thresholds).float()    # (B, T, K-1)
        # 패딩 위치 마스킹
        binary_targets[~valid] = 0.0

        # BCE with logits — reduction='none' 으로 유효 위치만 평균
        loss_map = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )  # (B, T, K-1)

        mask = valid.unsqueeze(-1).expand_as(loss_map).float()
        loss = (loss_map * mask).sum() / (mask.sum() + 1e-8)
        return loss


# ────────────────────────────────────────────────────────────────────────────
# 2.  Hierarchical Cross-Entropy Loss
# ────────────────────────────────────────────────────────────────────────────
def _focal_weight(probs: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Focal Loss 가중치 (1 - p_t)^gamma 계산.
    probs  : (B, C) softmax 확률
    targets: (B,)  long 레이블
    반환값 : (B,) 스칼라 가중치
    """
    pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)   # (B,)
    return (1.0 - pt) ** gamma


class HierarchicalCrossEntropyLoss(nn.Module):
    """
    Spec §3.2 계층적 손실.

    forward 인자:
      super_logits   : (B, 2)
      normal_logits  : (B, 3)
      phishing_logits: (B, 2)
      targets        : (B,)  전체 5-class 레이블

    반환: scalar 손실
    """

    def __init__(
        self,
        hce_lambda: float = LOSS_CONFIG["HCE_LAMBDA"],
        focal_gamma: float = LOSS_CONFIG["FOCAL_GAMMA"],
    ):
        super().__init__()
        self.hce_lambda = hce_lambda
        self.focal_gamma = focal_gamma

    def forward(
        self,
        super_logits: torch.Tensor,
        normal_logits: torch.Tensor,
        phishing_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        device = targets.device
        B = targets.size(0)

        # ── 상위 클래스 레이블 생성 ─────────────────────────────────────
        super_targets = torch.tensor(
            [SUPERCLASS_MAP[int(t.item())] for t in targets],
            dtype=torch.long,
            device=device,
        )  # (B,)

        # ── 상위 클래스 손실 (L_super) ──────────────────────────────────
        if self.focal_gamma > 0:
            super_probs = torch.softmax(super_logits, dim=-1)
            fw = _focal_weight(super_probs, super_targets, self.focal_gamma)
            loss_super_per = F.cross_entropy(super_logits, super_targets, reduction="none")
            loss_super = (fw * loss_super_per).mean()
        else:
            loss_super = F.cross_entropy(super_logits, super_targets)

        # ── 일반 하위 클래스 손실 ────────────────────────────────────────
        normal_mask = super_targets == 0                         # (B,)
        if normal_mask.any():
            n_logits = normal_logits[normal_mask]               # (N, 3)
            n_targets = targets[normal_mask]                     # (N,) ∈ {0,1,2}
            if self.focal_gamma > 0:
                n_probs = torch.softmax(n_logits, dim=-1)
                fw_n = _focal_weight(n_probs, n_targets, self.focal_gamma)
                loss_normal = (fw_n * F.cross_entropy(n_logits, n_targets, reduction="none")).mean()
            else:
                loss_normal = F.cross_entropy(n_logits, n_targets)
        else:
            loss_normal = torch.tensor(0.0, device=device)

        # ── 피싱 하위 클래스 손실 ────────────────────────────────────────
        phishing_mask = super_targets == 1                       # (B,)
        if phishing_mask.any():
            p_logits = phishing_logits[phishing_mask]           # (P, 2)
            # 원래 레이블 {3,4} → 피싱 내 {0,1}
            p_targets = targets[phishing_mask] - 3              # (P,)
            if self.focal_gamma > 0:
                p_probs = torch.softmax(p_logits, dim=-1)
                fw_p = _focal_weight(p_probs, p_targets, self.focal_gamma)
                loss_phishing = (fw_p * F.cross_entropy(p_logits, p_targets, reduction="none")).mean()
            else:
                loss_phishing = F.cross_entropy(p_logits, p_targets)
        else:
            loss_phishing = torch.tensor(0.0, device=device)

        # ── 통합 ─────────────────────────────────────────────────────────
        # L_main = L_super + λ * (L_normal + L_phishing)
        loss_sub = loss_normal + loss_phishing
        loss_main = loss_super + self.hce_lambda * loss_sub
        return loss_main
