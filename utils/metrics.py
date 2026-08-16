"""
Evaluation metrics for Foreground Protection / Dimming Soft Mask Prediction.

Metric groups
-------------
1. **Binary segmentation**: IoU, Dice, Precision, Recall  (threshold = 0.5)
2. **Soft-mask quality**: MAE, MSE  (pred prob vs soft target)
3. **Foreground protection**: mean protection, under-protection error,
   under-protection rate @0.9
4. **Far-background leakage**: mean prediction where soft_target == 0

All metrics operate on batched tensors and return Python floats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch


@dataclass
class MetricAccumulator:
    """Accumulates metric sums and counts across batches.

    Call ``update()`` for each batch, then ``compute()`` for final averages.
    """

    # Binary segmentation (accumulated over pixels)
    _tp: int = 0
    _fp: int = 0
    _fn: int = 0
    _tn: int = 0

    # Soft mask
    _soft_mae_sum: float = 0.0
    _soft_mse_sum: float = 0.0
    _soft_count: int = 0

    # Foreground protection
    _fg_prob_sum: float = 0.0
    _fg_under_sum: float = 0.0
    _fg_under_09_count: int = 0
    _fg_pixel_count: int = 0

    # Far-background leakage
    _far_bg_prob_sum: float = 0.0
    _far_bg_pixel_count: int = 0

    def reset(self) -> None:
        """Reset all accumulators."""
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self._soft_mae_sum = 0.0
        self._soft_mse_sum = 0.0
        self._soft_count = 0
        self._fg_prob_sum = 0.0
        self._fg_under_sum = 0.0
        self._fg_under_09_count = 0
        self._fg_pixel_count = 0
        self._far_bg_prob_sum = 0.0
        self._far_bg_pixel_count = 0

    @torch.no_grad()
    def update(
        self,
        prob: torch.Tensor,
        soft_target: torch.Tensor,
        binary_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Update accumulators with a batch of predictions.

        Parameters
        ----------
        prob : Tensor [B, 1, H, W]
            Sigmoid-activated model prediction.
        soft_target : Tensor [B, 1, H, W]
            Soft dimming protection target.
        binary_mask : Tensor [B, 1, H, W]
            Original binary foreground GT.
        threshold : float
            Binarization threshold for segmentation metrics.
        """
        # --- Binary segmentation metrics ---
        pred_binary = (prob >= threshold).float()
        gt_binary = binary_mask.float()

        self._tp += int(((pred_binary == 1) & (gt_binary == 1)).sum().item())
        self._fp += int(((pred_binary == 1) & (gt_binary == 0)).sum().item())
        self._fn += int(((pred_binary == 0) & (gt_binary == 1)).sum().item())
        self._tn += int(((pred_binary == 0) & (gt_binary == 0)).sum().item())

        # --- Soft-mask metrics ---
        diff = prob - soft_target
        self._soft_mae_sum += float(diff.abs().sum().item())
        self._soft_mse_sum += float((diff ** 2).sum().item())
        self._soft_count += int(prob.numel())

        # --- Foreground protection metrics ---
        fg_mask = (binary_mask == 1)
        fg_count = int(fg_mask.sum().item())
        if fg_count > 0:
            fg_probs = prob[fg_mask]
            self._fg_prob_sum += float(fg_probs.sum().item())
            self._fg_under_sum += float((1.0 - fg_probs).sum().item())
            self._fg_under_09_count += int((fg_probs < 0.9).sum().item())
            self._fg_pixel_count += fg_count

        # --- Far-background leakage ---
        # Only pixels where soft_target == 0 (NOT transition region)
        far_bg_mask = (soft_target == 0)
        far_bg_count = int(far_bg_mask.sum().item())
        if far_bg_count > 0:
            self._far_bg_prob_sum += float(prob[far_bg_mask].sum().item())
            self._far_bg_pixel_count += far_bg_count

    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated values.

        Returns
        -------
        dict with all metric names and values.
        """
        eps = 1e-8
        results: Dict[str, float] = {}

        # Binary segmentation
        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        fg_iou = tp / (tp + fp + fn + eps)
        bg_iou = tn / (tn + fp + fn + eps)
        results["fg_iou"] = fg_iou
        results["bg_iou"] = bg_iou
        results["miou"] = (fg_iou + bg_iou) / 2.0
        results["dice"] = (2 * tp) / (2 * tp + fp + fn + eps)
        results["precision"] = tp / (tp + fp + eps)
        results["recall"] = tp / (tp + fn + eps)

        # Soft mask
        if self._soft_count > 0:
            results["soft_mae"] = self._soft_mae_sum / self._soft_count
            results["soft_mse"] = self._soft_mse_sum / self._soft_count
        else:
            results["soft_mae"] = 0.0
            results["soft_mse"] = 0.0

        # Foreground protection
        if self._fg_pixel_count > 0:
            results["fg_mean_protection"] = (
                self._fg_prob_sum / self._fg_pixel_count
            )
            results["fg_under_protection_error"] = (
                self._fg_under_sum / self._fg_pixel_count
            )
            results["fg_under_protection_rate_09"] = (
                self._fg_under_09_count / self._fg_pixel_count
            )
        else:
            results["fg_mean_protection"] = 1.0
            results["fg_under_protection_error"] = 0.0
            results["fg_under_protection_rate_09"] = 0.0

        # Far-background leakage
        if self._far_bg_pixel_count > 0:
            results["far_bg_leakage"] = (
                self._far_bg_prob_sum / self._far_bg_pixel_count
            )
        else:
            results["far_bg_leakage"] = 0.0

        return results


class MulticlassMetricAccumulator:
    """Confusion matrix based metric accumulator for multiclass semantic segmentation (COCO-Stuff / ADE20K).

    Computes:
    - Mean IoU (mIoU) across all present classes
    - Pixel Accuracy (pixel_acc)
    - Mean Dice coefficient
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    @torch.no_grad()
    def update(
        self,
        pred_labels: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> None:
        """Update confusion matrix with a batch.

        Parameters
        ----------
        pred_labels : Tensor [B, H, W] or [B, 1, H, W]
            Predicted integer class IDs (e.g. from argmax).
        gt_labels : Tensor [B, H, W] or [B, 1, H, W]
            Ground truth integer class IDs.
        """
        if pred_labels.dim() == 4 and pred_labels.shape[1] == 1:
            pred_labels = pred_labels.squeeze(1)
        if gt_labels.dim() == 4 and gt_labels.shape[1] == 1:
            gt_labels = gt_labels.squeeze(1)

        preds = pred_labels.detach().cpu().numpy().flatten()
        gts = gt_labels.detach().cpu().numpy().flatten()

        # Mask out ignore index and out-of-range labels
        valid_mask = (gts != self.ignore_index) & (gts >= 0) & (gts < self.num_classes)
        valid_preds = preds[valid_mask]
        valid_gts = gts[valid_mask]

        if len(valid_gts) > 0:
            # 2D bincount is much faster than per-cell loop
            idx = valid_gts * self.num_classes + valid_preds
            counts = np.bincount(idx, minlength=self.num_classes ** 2)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self, eps: float = 1e-6) -> Dict[str, float]:
        """Compute mIoU, Pixel Accuracy, and Mean Dice."""
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        union = tp + fp + fn
        # Only evaluate classes that appear in GT (where fn + tp > 0)
        present_mask = (cm.sum(axis=1) > 0)

        iou = np.zeros(self.num_classes, dtype=np.float64)
        iou[present_mask] = tp[present_mask] / (union[present_mask] + eps)
        miou = float(np.mean(iou[present_mask])) if np.any(present_mask) else 0.0

        dice = np.zeros(self.num_classes, dtype=np.float64)
        denom = 2 * tp[present_mask] + fp[present_mask] + fn[present_mask]
        dice[present_mask] = (2 * tp[present_mask]) / (denom + eps)
        mean_dice = float(np.mean(dice[present_mask])) if np.any(present_mask) else 0.0

        total_correct = int(tp.sum())
        total_pixels = int(cm.sum())
        pixel_acc = float(total_correct / max(total_pixels, 1))

        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "dice": mean_dice,
        }


def format_metrics(metrics: Dict[str, float], threshold: float = 0.5) -> str:
    """Format metrics dict into a human-readable multi-line string."""
    lines = []
    lines.append("=" * 50)
    lines.append("Evaluation Metrics")
    lines.append("=" * 50)

    # Binary segmentation
    lines.append(f"\n--- Binary Segmentation (threshold={threshold}) ---")
    lines.append(f"  Foreground IoU     : {metrics.get('fg_iou', 0):.4f}")
    lines.append(f"  Mean IoU (mIoU)    : {metrics.get('miou', 0):.4f}")
    lines.append(f"  Dice               : {metrics.get('dice', 0):.4f}")
    lines.append(f"  Precision          : {metrics.get('precision', 0):.4f}")
    lines.append(f"  Recall             : {metrics.get('recall', 0):.4f}")

    # Soft mask
    lines.append("\n--- Soft Mask Quality ---")
    lines.append(f"  MAE                : {metrics.get('soft_mae', 0):.4f}")
    lines.append(f"  MSE                : {metrics.get('soft_mse', 0):.6f}")

    # Foreground protection
    lines.append("\n--- Foreground Protection ---")
    lines.append(
        f"  Mean protection    : {metrics.get('fg_mean_protection', 0):.4f}"
    )
    lines.append(
        f"  Under-protect err  : {metrics.get('fg_under_protection_error', 0):.4f}"
    )
    lines.append(
        f"  Under-protect @0.9 : {metrics.get('fg_under_protection_rate_09', 0):.4f}"
    )

    # Far-background leakage
    lines.append("\n--- Far-Background Leakage ---")
    lines.append(
        f"  Mean leakage       : {metrics.get('far_bg_leakage', 0):.4f}"
    )

    lines.append("=" * 50)
    return "\n".join(lines)
