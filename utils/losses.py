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

from typing import Dict, Optional, Union

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
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        prob : Tensor [B, 1, H, W]
            Sigmoid-activated prediction (0–1).
        binary_mask : Tensor [B, 1, H, W]
            Original binary foreground GT (0 or 1).
        weight : Tensor [B, 1, H, W], optional
            Pixel-wise loss weight tensor.
        """
        under_protection = (1.0 - prob) * binary_mask
        if weight is not None:
            under_protection = under_protection * weight
            fg_count = (binary_mask * weight).sum() + self.eps
        else:
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
        self.bce_loss_none = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss_none = nn.L1Loss(reduction="none")
        self.fg_protect_loss = ForegroundProtectionLoss()

    def forward(
        self,
        logits: torch.Tensor,
        soft_target: torch.Tensor,
        binary_mask: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
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
        weight : Tensor [B, 1, H, W], optional
            Pixel-wise loss weight tensor.

        Returns
        -------
        dict with keys: "total", "bce", "l1", "protect"
        """
        if weight is not None:
            # Apply pixel-wise weights (Option 2: Edge-Masked Loss)
            loss_bce = self.bce_loss_none(logits, soft_target)
            loss_bce = (loss_bce * weight).mean()

            prob = torch.sigmoid(logits)
            loss_l1 = self.l1_loss_none(prob, soft_target)
            loss_l1 = (loss_l1 * weight).mean()

            loss_protect = self.fg_protect_loss(prob, binary_mask, weight=weight)
        else:
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
        weight: Optional[torch.Tensor] = None,
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


class DualHeadLoss(nn.Module):
    """Composite loss for Dual-Head (Coarse + Fine) segmentation models.

    Loss Formulation
    ----------------
    L_total = L_fine + lambda_coarse * L_coarse

    - If num_classes == 1 (binary foreground dimming):
        L_fine   = DimmingLoss(fine_logits, soft_target, binary_mask)
        L_coarse = DimmingLoss(coarse_logits, soft_target, binary_mask)
    - If num_classes > 1 (multiclass):
        L_fine   = MulticlassCrossEntropyLoss(fine_logits, soft_target)
        L_coarse = MulticlassCrossEntropyLoss(coarse_logits, soft_target)

    Parameters
    ----------
    base_loss : nn.Module
        Underlying loss (DimmingLoss or MulticlassCrossEntropyLoss).
    lambda_coarse : float
        Weight for coarse auxiliary loss (default: 0.5).
    coarse_edge_mask_kernel : int
        Edge mask kernel size for CoarseHead loss (default: 15).
    coarse_target_dilation_kernel : int
        Target dilation kernel size for CoarseHead loss (default: 15).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        lambda_coarse: float = 0.5,
        coarse_only_epochs: int = 0,
        coarse_edge_mask_kernel: int = 15,
        coarse_target_dilation_kernel: int = 15,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.lambda_coarse = lambda_coarse
        self.coarse_only_epochs = coarse_only_epochs
        self.coarse_edge_mask_kernel = coarse_edge_mask_kernel
        self.coarse_target_dilation_kernel = coarse_target_dilation_kernel
        self.current_epoch = 0

    def forward(
        self,
        pred: torch.Tensor | Dict[str, torch.Tensor],
        soft_target: torch.Tensor,
        binary_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(pred, dict):
            fine_logits = pred["fine_logits"]
            coarse_logits = pred["coarse_logits"]

            # Compute dilated targets for coarse head if enabled
            coarse_soft_target = soft_target
            coarse_binary_mask = binary_mask
            if self.coarse_target_dilation_kernel > 0:
                pad_dil = self.coarse_target_dilation_kernel // 2
                coarse_soft_target = F.max_pool2d(soft_target, kernel_size=self.coarse_target_dilation_kernel, stride=1, padding=pad_dil)
                if binary_mask is not None:
                    coarse_binary_mask = F.max_pool2d(binary_mask, kernel_size=self.coarse_target_dilation_kernel, stride=1, padding=pad_dil)

            # Compute edge mask for coarse head if enabled
            coarse_weight = None
            if self.coarse_edge_mask_kernel > 0 and coarse_binary_mask is not None:
                pad = self.coarse_edge_mask_kernel // 2
                dilated = F.max_pool2d(coarse_binary_mask, kernel_size=self.coarse_edge_mask_kernel, stride=1, padding=pad)
                eroded = -F.max_pool2d(-coarse_binary_mask, kernel_size=self.coarse_edge_mask_kernel, stride=1, padding=pad)
                edge_mask = torch.clamp(dilated - eroded, 0.0, 1.0)
                coarse_weight = 1.0 - edge_mask

            fine_losses = self.base_loss(fine_logits, soft_target, binary_mask)
            coarse_losses = self.base_loss(coarse_logits, coarse_soft_target, coarse_binary_mask, weight=coarse_weight)

            if self.training and self.current_epoch < self.coarse_only_epochs:
                total = self.lambda_coarse * coarse_losses["total"]
                fine_weight = 0.0
                coarse_train_weight = 1.0
            else:
                total = fine_losses["total"]
                fine_weight = 1.0
                coarse_train_weight = 0.0

            return {
                "total": total,
                "fine_total": fine_weight * fine_losses["total"],
                "coarse_total": coarse_train_weight * coarse_losses["total"],
                "bce": fine_weight * fine_losses["bce"],
                "l1": fine_weight * fine_losses["l1"],
                "protect": fine_weight * fine_losses["protect"],
                "coarse_bce": coarse_train_weight * coarse_losses["bce"],
            }
        else:
            return self.base_loss(pred, soft_target, binary_mask)


def build_criterion(
    num_classes: int = 1,
    lambda_bce: float = 1.0,
    lambda_l1: float = 1.0,
    lambda_protect: float = 2.0,
    ignore_index: int = 255,
    is_dual_head: bool = False,
    lambda_coarse: float = 0.5,
    coarse_only_epochs: int = 0,
    coarse_edge_mask_kernel: int = 15,
    coarse_target_dilation_kernel: int = 15,
) -> nn.Module:
    """Build appropriate loss criterion based on num_classes and model type.

    - num_classes == 1: DimmingLoss (BCE + L1 + Foreground Protection)
    - num_classes > 1: MulticlassCrossEntropyLoss (CrossEntropy with ignore_index=255)
    - is_dual_head: Wraps with DualHeadLoss (L_fine + lambda_coarse * L_coarse)
    """
    if num_classes > 1:
        base = MulticlassCrossEntropyLoss(ignore_index=ignore_index)
    else:
        base = DimmingLoss(
            lambda_bce=lambda_bce,
            lambda_l1=lambda_l1,
            lambda_protect=lambda_protect,
        )

    if is_dual_head:
        return DualHeadLoss(
            base_loss=base,
            lambda_coarse=lambda_coarse,
            coarse_only_epochs=coarse_only_epochs,
            coarse_edge_mask_kernel=coarse_edge_mask_kernel,
            coarse_target_dilation_kernel=coarse_target_dilation_kernel,
        )
    return base

