"""
koelectra_mamba_short_window_freeze_init / losses.py

koelectra_mamba_short_window 과 동일한 손실 함수.
1. OrdinalRegressionLoss
2. HierarchicalCrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOSS_CONFIG, SUPERCLASS_MAP


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, Km1 = logits.shape

        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0

        y_exp = targets.unsqueeze(-1).expand(B, T, Km1)
        thresholds = torch.arange(Km1, device=logits.device).view(1, 1, Km1)
        binary_targets = (y_exp > thresholds).float()
        binary_targets[~valid] = 0.0

        loss_map = F.binary_cross_entropy_with_logits(
            logits, binary_targets, reduction="none"
        )

        mask = valid.unsqueeze(-1).expand_as(loss_map).float()
        loss = (loss_map * mask).sum() / (mask.sum() + 1e-8)
        return loss


def _focal_weight(probs: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)
    return (1.0 - pt) ** gamma


class HierarchicalCrossEntropyLoss(nn.Module):
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

        super_targets = torch.tensor(
            [SUPERCLASS_MAP[int(t.item())] for t in targets],
            dtype=torch.long,
            device=device,
        )

        if self.focal_gamma > 0:
            super_probs = torch.softmax(super_logits, dim=-1)
            fw = _focal_weight(super_probs, super_targets, self.focal_gamma)
            loss_super = (fw * F.cross_entropy(super_logits, super_targets, reduction="none")).mean()
        else:
            loss_super = F.cross_entropy(super_logits, super_targets)

        normal_mask = super_targets == 0
        if normal_mask.any():
            n_logits = normal_logits[normal_mask]
            n_targets = targets[normal_mask]
            if self.focal_gamma > 0:
                n_probs = torch.softmax(n_logits, dim=-1)
                fw_n = _focal_weight(n_probs, n_targets, self.focal_gamma)
                loss_normal = (fw_n * F.cross_entropy(n_logits, n_targets, reduction="none")).mean()
            else:
                loss_normal = F.cross_entropy(n_logits, n_targets)
        else:
            loss_normal = torch.tensor(0.0, device=device)

        phishing_mask = super_targets == 1
        if phishing_mask.any():
            p_logits = phishing_logits[phishing_mask]
            p_targets = targets[phishing_mask] - 3
            if self.focal_gamma > 0:
                p_probs = torch.softmax(p_logits, dim=-1)
                fw_p = _focal_weight(p_probs, p_targets, self.focal_gamma)
                loss_phishing = (fw_p * F.cross_entropy(p_logits, p_targets, reduction="none")).mean()
            else:
                loss_phishing = F.cross_entropy(p_logits, p_targets)
        else:
            loss_phishing = torch.tensor(0.0, device=device)

        return loss_super + self.hce_lambda * (loss_normal + loss_phishing)
