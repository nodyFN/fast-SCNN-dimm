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


def save_side_by_side(
    save_path: Path | str,
    image: np.ndarray,
    binary_mask: np.ndarray,
    soft_mask: np.ndarray,
    min_brightness: float = 0.5,
    pred_mask: Optional[np.ndarray] = None,
) -> None:
    """Save a multi-panel visualization.

    Panels (left to right):
        1. Original image
        2. Binary mask (or pred binary)
        3. Soft protection mask (heatmap)
        4. Brightness map (heatmap)
        5. Dimmed preview

    Parameters
    ----------
    save_path : Path
        Output file path.
    image : ndarray [H, W, C], uint8, BGR
    binary_mask : ndarray [H, W], float32 or uint8
    soft_mask : ndarray [H, W], float32 in [0, 1]
    min_brightness : float
    pred_mask : ndarray [H, W], float32, optional
        If provided, used instead of soft_mask for the prediction panel.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = image.shape[:2]

    # Panel 1: Original
    panel_orig = image.copy()

    # Panel 2: Binary mask visualization
    binary_vis = (np.clip(binary_mask, 0, 1) * 255).astype(np.uint8)
    panel_binary = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)

    # Panel 3: Soft mask heatmap
    display_mask = pred_mask if pred_mask is not None else soft_mask
    panel_soft = mask_to_heatmap(display_mask)

    # Panel 4: Brightness map
    brightness = protection_to_brightness(display_mask, min_brightness)
    panel_brightness = mask_to_heatmap(brightness)

    # Panel 5: Dimmed preview
    panel_dimmed = create_dimmed_preview(image, display_mask, min_brightness)

    # Concatenate horizontally
    row = np.concatenate(
        [panel_orig, panel_binary, panel_soft, panel_brightness, panel_dimmed],
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
    """Save training-time validation visualization.

    Creates a grid of side-by-side panels for multiple samples.

    Parameters
    ----------
    images : Tensor [B, 3, H, W], normalized
    binary_masks : Tensor [B, 1, H, W]
    soft_targets : Tensor [B, 1, H, W]
    predictions : Tensor [B, 1, H, W], probability (after sigmoid)
    num_samples : int
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

        # Panels: orig | GT binary | GT soft | pred soft | dimmed preview
        panel_orig = img.copy()
        panel_binary = cv2.cvtColor(
            (np.clip(binary, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        panel_gt_soft = mask_to_heatmap(soft)
        panel_pred_soft = mask_to_heatmap(pred)
        panel_dimmed = create_dimmed_preview(img, pred, min_brightness)

        row = np.concatenate(
            [panel_orig, panel_binary, panel_gt_soft, panel_pred_soft, panel_dimmed],
            axis=1,
        )
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(save_path), grid)
