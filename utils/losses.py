"""
Loss functions for Foreground Protection / Dimming Soft Mask Prediction.

Total loss
----------
::

    L_total = λ_bce * L_bce  +  λ_l1 * L_l1  +  λ_protect * L_protect

- **L_bce**: BCEWithLogitsLoss(logits, soft_target)
- **L_l1**: L1(sigmoid(logits), soft_target)
- **L_protect**: foreground under-protection penalty

[PROJECT DECISION] No Dice, Focal, Tversky, Edge, Sobel, KD, or matting loss
in V1.  Ablation in future experiments.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundProtectionLoss(nn.Module):
    """Penalise under-protection of foreground pixels.

    Computes:
        L = sum((1 - prob) * binary_mask) / (sum(binary_mask) + eps)

    Only original binary foreground (binary_mask == 1) pixels contribute.
    If the sample has no foreground pixels, the loss is 0 (avoids NaN).

    This asymmetric loss embodies the product requirement: foreground
    under-protection should be penalised more heavily than background
    over-protection.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prob : Tensor [B, 1, H, W]
            Sigmoid-activated prediction (0–1).
        binary_mask : Tensor [B, 1, H, W]
            Original binary foreground GT (0 or 1).
        """
        under_protection = (1.0 - prob) * binary_mask
        fg_count = binary_mask.sum() + self.eps
        return under_protection.sum() / fg_count


class DimmingLoss(nn.Module):
    """Composite loss for dimming soft mask prediction.

    L_total = λ_bce * BCEWithLogits(logits, soft_target)
            + λ_l1  * L1(sigmoid(logits), soft_target)
            + λ_protect * ForegroundProtectionLoss(sigmoid(logits), binary_mask)

    Parameters
    ----------
    lambda_bce : float
        Weight for BCE loss.  [PROJECT DECISION] default = 1.0.
    lambda_l1 : float
        Weight for L1 loss.  [PROJECT DECISION] default = 1.0.
    lambda_protect : float
        Weight for foreground protection loss.  [PROJECT DECISION] default = 2.0.
    """

    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_l1: float = 1.0,
        lambda_protect: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_l1 = lambda_l1
        self.lambda_protect = lambda_protect

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.fg_protect_loss = ForegroundProtectionLoss()

    def forward(
        self,
        logits: torch.Tensor,
        soft_target: torch.Tensor,
        binary_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        logits : Tensor [B, 1, H, W]
            Raw model output (before sigmoid).
        soft_target : Tensor [B, 1, H, W]
            Soft dimming protection target, values in [0, 1].
        binary_mask : Tensor [B, 1, H, W]
            Original binary foreground GT (0 or 1).

        Returns
        -------
        dict with keys: "total", "bce", "l1", "protect"
        """
        # BCE — operates on logits directly (numerically stable)
        loss_bce = self.bce_loss(logits, soft_target)

        # L1 — operates on probability space
        prob = torch.sigmoid(logits)
        loss_l1 = self.l1_loss(prob, soft_target)

        # Foreground protection — only on binary foreground
        loss_protect = self.fg_protect_loss(prob, binary_mask)

        # Weighted sum
        total = (
            self.lambda_bce * loss_bce
            + self.lambda_l1 * loss_l1
            + self.lambda_protect * loss_protect
        )

        return {
            "total": total,
            "bce": loss_bce,
            "l1": loss_l1,
            "protect": loss_protect,
        }


class MulticlassCrossEntropyLoss(nn.Module):
    """Standard CrossEntropyLoss for multiclass semantic segmentation pretraining (COCO-Stuff / ADE20K).

    Parameters
    ----------
    ignore_index : int
        Class label index to ignore (default: 255).
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        soft_target: torch.Tensor,
        binary_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        logits : Tensor [B, num_classes, H, W]
        soft_target : Tensor [B, 1, H, W] or [B, H, W]
            Integer class indices.
        binary_mask : optional Tensor, ignored in multiclass mode.

        Returns
        -------
        dict with keys: "total", "ce", "bce", "l1", "protect"
        """
        targets = soft_target
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        targets = targets.long()

        loss_ce = self.ce_loss(logits, targets)
        zero = torch.tensor(0.0, device=logits.device)
        return {
            "total": loss_ce,
            "ce": loss_ce,
            "bce": loss_ce,
            "l1": zero,
            "protect": zero,
        }


def build_criterion(
    num_classes: int = 1,
    lambda_bce: float = 1.0,
    lambda_l1: float = 1.0,
    lambda_protect: float = 2.0,
    ignore_index: int = 255,
) -> nn.Module:
    """Build appropriate loss criterion based on num_classes.

    - num_classes == 1: DimmingLoss (BCE + L1 + Foreground Protection)
    - num_classes > 1: MulticlassCrossEntropyLoss (CrossEntropy with ignore_index=255)
    """
    if num_classes > 1:
        return MulticlassCrossEntropyLoss(ignore_index=ignore_index)
    return DimmingLoss(
        lambda_bce=lambda_bce,
        lambda_l1=lambda_l1,
        lambda_protect=lambda_protect,
    )
