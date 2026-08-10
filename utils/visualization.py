"""
Visualization utilities for Foreground Protection / Dimming Soft Mask Prediction.

Provides:
- ``protection_to_brightness``: Simulated brightness mapping from protection mask.
- ``create_dimmed_preview``: Apply simulated dimming to an image.
- ``save_side_by_side``: Save multi-panel comparison (original, masks, dimmed preview).
- ``save_training_visualization``: Save training-time validation visualizations.

[PROJECT DECISION] The brightness mapping is a simulation, not a real TV power model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# ImageNet denormalization
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(
    tensor: torch.Tensor | np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """Convert a normalized [C, H, W] tensor back to uint8 [H, W, C] BGR image.

    Parameters
    ----------
    tensor : Tensor or ndarray
        Shape [C, H, W], float, normalized with ImageNet mean/std.

    Returns
    -------
    image : ndarray  [H, W, C], uint8, BGR
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor.copy()

    # [C, H, W] → [H, W, C]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)

    # Denormalize
    arr = arr * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    # RGB → BGR for OpenCV
    if arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return arr


# ---------------------------------------------------------------------------
# Brightness / dimming simulation
# ---------------------------------------------------------------------------


def protection_to_brightness(
    mask: np.ndarray,
    min_brightness: float = 0.5,
) -> np.ndarray:
    """Convert protection mask to brightness multiplier.

    [PROJECT DECISION] Simulated formula:
        D = D_min + (1 - D_min) * M

    Examples (D_min = 0.5):
        M=1.0 → brightness 100%
        M=0.8 → 90%
        M=0.5 → 75%
        M=0.0 → 50%

    This is a visualization aid, NOT a real TV power model.

    Parameters
    ----------
    mask : ndarray [H, W], float32, values in [0, 1]
    min_brightness : float
        Minimum brightness for fully dimmable background.

    Returns
    -------
    brightness : ndarray [H, W], float32, values in [min_brightness, 1.0]
    """
    return min_brightness + (1.0 - min_brightness) * mask


def create_dimmed_preview(
    image: np.ndarray,
    mask: np.ndarray,
    min_brightness: float = 0.5,
) -> np.ndarray:
    """Apply simulated dimming to an image.

    Parameters
    ----------
    image : ndarray [H, W, C], uint8
    mask : ndarray [H, W], float32, values in [0, 1]
    min_brightness : float

    Returns
    -------
    dimmed : ndarray [H, W, C], uint8
    """
    brightness = protection_to_brightness(mask, min_brightness)
    # Apply per-pixel brightness scaling
    dimmed = image.astype(np.float32) * brightness[:, :, np.newaxis]
    return np.clip(dimmed, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization panels
# ---------------------------------------------------------------------------


def mask_to_heatmap(mask: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Convert a [0, 1] mask to a BGR heatmap image.

    Parameters
    ----------
    mask : ndarray [H, W], float32

    Returns
    -------
    heatmap : ndarray [H, W, 3], uint8, BGR
    """
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(mask_uint8, colormap)


def add_panel_label(
    panel: np.ndarray,
    label: str,
    banner_height: int = 22,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a header banner with a text description at the top of a panel."""
    w = panel.shape[1]
    banner = np.full((banner_height, w, 3), bg_color, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x = max((w - text_w) // 2, 4)
    y = (banner_height + text_h) // 2
    cv2.putText(banner, label, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return np.vstack([banner, panel])


def save_side_by_side(
    save_path: Path | str,
    image: np.ndarray,
    binary_mask: np.ndarray,
    soft_mask: np.ndarray,
    min_brightness: float = 0.5,
    pred_mask: Optional[np.ndarray] = None,
) -> None:
    """Save a multi-panel visualization with labeled headers.

    Panels (left to right):
        1. Original image
        2. Binary mask
        3. Soft protection mask (heatmap)
        4. Predicted soft mask (grayscale)
        5. Brightness map (heatmap)
        6. Dimmed preview
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Panel 1: Original
    panel_orig = add_panel_label(image.copy(), "Original RGB")

    # Panel 2: Binary mask visualization
    binary_vis = (np.clip(binary_mask, 0, 1) * 255).astype(np.uint8)
    panel_binary = add_panel_label(cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR), "GT Binary")

    # Panel 3: Soft mask heatmap
    display_mask = pred_mask if pred_mask is not None else soft_mask
    panel_soft = add_panel_label(mask_to_heatmap(soft_mask), "GT Soft Target")

    # Panel 4: Pred soft grayscale
    pred_gray = (np.clip(display_mask, 0, 1) * 255).astype(np.uint8)
    panel_pred_gray = add_panel_label(cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR), "Pred Soft (Gray)")

    # Panel 5: Pred soft heatmap
    panel_pred_heat = add_panel_label(mask_to_heatmap(display_mask), "Pred Soft (Heatmap)")

    # Panel 6: Dimmed preview
    dimmed = create_dimmed_preview(image, display_mask, min_brightness)
    panel_dimmed = add_panel_label(dimmed, "Dimmed Preview")

    # Concatenate horizontally
    row = np.concatenate(
        [panel_orig, panel_binary, panel_soft, panel_pred_gray, panel_pred_heat, panel_dimmed],
        axis=1,
    )

    cv2.imwrite(str(save_path), row)


def save_training_visualization(
    save_path: Path | str,
    images: torch.Tensor,
    binary_masks: torch.Tensor,
    soft_targets: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 4,
    min_brightness: float = 0.5,
) -> None:
    """Save training-time validation visualization with labeled headers.

    Creates a grid of side-by-side labeled panels for multiple samples.
    Panels per sample:
        1. Original RGB
        2. GT Binary
        3. GT Soft Target
        4. Pred Soft (Gray)
        5. Pred Soft (Heatmap)
        6. Dimmed Preview

    Parameters
    ----------
    images : Tensor [N, 3, H, W], normalized
    binary_masks : Tensor [N, 1, H, W]
    soft_targets : Tensor [N, 1, H, W]
    predictions : Tensor [N, 1, H, W], probability (after sigmoid)
    num_samples : int
    min_brightness : float
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, images.size(0))
    rows = []

    for i in range(n):
        img = denormalize_image(images[i])
        binary = binary_masks[i, 0].detach().cpu().numpy()
        soft = soft_targets[i, 0].detach().cpu().numpy()
        pred = predictions[i, 0].detach().cpu().numpy()

        # Build each labeled panel
        panel_orig = add_panel_label(img.copy(), "Original RGB")

        binary_vis = (np.clip(binary, 0, 1) * 255).astype(np.uint8)
        panel_binary = add_panel_label(
            cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR), "GT Binary"
        )

        panel_gt_soft = add_panel_label(mask_to_heatmap(soft), "GT Soft Target")

        pred_gray = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        panel_pred_gray = add_panel_label(
            cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR), "Pred Soft (Gray)"
        )

        panel_pred_heat = add_panel_label(mask_to_heatmap(pred), "Pred Soft (Heatmap)")

        dimmed = create_dimmed_preview(img, pred, min_brightness)
        panel_dimmed = add_panel_label(dimmed, "Dimmed Preview")

        row = np.concatenate(
            [panel_orig, panel_binary, panel_gt_soft, panel_pred_gray, panel_pred_heat, panel_dimmed],
            axis=1,
        )
        rows.append(row)

    if rows:
        grid = np.concatenate(rows, axis=0)
        cv2.imwrite(str(save_path), grid)


def plot_training_curves(
    history: Dict[str, list],
    save_path: Path | str,
) -> None:
    """Plot training/validation loss and metric curves over epochs.

    Parameters
    ----------
    history : dict
        Dictionary containing history lists:
        - "epoch"
        - "train_loss", "val_loss"
        - "fg_iou", "miou", "dice"
        - "fg_mean_protection", "soft_mae", "far_bg_leakage"
    save_path : Path or str
        Destination path for the plotted figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    if not epochs:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Loss Curves
    ax1 = axes[0]
    if "train_loss" in history and history["train_loss"]:
        ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_title("Loss Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Segmentation Metrics (IoU, mIoU, Dice)
    ax2 = axes[1]
    if "fg_iou" in history and history["fg_iou"]:
        ax2.plot(epochs, history["fg_iou"], label="Foreground IoU", color="#2ca02c", linewidth=2)
    if "miou" in history and history["miou"]:
        ax2.plot(epochs, history["miou"], label="Mean IoU (mIoU)", color="#17becf", linewidth=2)
    if "dice" in history and history["dice"]:
        ax2.plot(epochs, history["dice"], label="Dice", color="#9467bd", linewidth=2, linestyle="--")
    ax2.set_title("Segmentation Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Subplot 3: Protection & Quality Metrics
    ax3 = axes[2]
    if "fg_mean_protection" in history and history["fg_mean_protection"]:
        ax3.plot(epochs, history["fg_mean_protection"], label="FG Protection (closer to 1)", color="#d62728", linewidth=2)
    if "soft_mae" in history and history["soft_mae"]:
        ax3.plot(epochs, history["soft_mae"], label="Soft MAE (lower is better)", color="#8c564b", linewidth=2)
    if "far_bg_leakage" in history and history["far_bg_leakage"]:
        ax3.plot(epochs, history["far_bg_leakage"], label="Far BG Leakage (lower is better)", color="#e377c2", linewidth=2, linestyle="--")
    ax3.set_title("Protection & Dimming Quality", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Value")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
