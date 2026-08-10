"""
Foreground-Preserving Soft Target Generator
============================================

Generates soft dimming protection targets from binary foreground masks.

Algorithm
---------
1. **Protection dilation**: Dilate binary foreground by ``protection_radius``
   using an ellipse structuring element.  All dilated pixels get M=1.

2. **Outward distance transform**: Compute distance from protected foreground
   boundary into background.  Uses ``cv2.distanceTransform`` on the
   *inverted* protected mask.

3. **Cosine feather**: For background pixels within ``transition_width``,
   apply cosine decay:  ``M(d) = 0.5 * (1 + cos(π * d / T))``

Properties
----------
- M=1 for all original foreground pixels (guaranteed by dilation).
- M=1 for dilated protection region.
- 0 < M < 1 for transition zone.
- M=0 for far background (distance ≥ transition_width).
- M ∈ [0, 1] everywhere.
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_soft_target(
    binary_mask: np.ndarray,
    protection_radius: int = 2,
    transition_width: int = 8,
    mode: str = "cosine",
) -> np.ndarray:
    """Generate a foreground-preserving soft dimming target.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary foreground mask, shape (H, W), values in {0, 1}, dtype uint8
        or float32.
    protection_radius : int
        Dilation radius.  radius=2 → ~5×5 ellipse structuring element.
        [PROJECT DECISION] default = 2.
    transition_width : int
        Width (in pixels) of the cosine feather transition zone.
        [PROJECT DECISION] default = 8.
    mode : str
        Feather mode.  Currently supported: "cosine".
        "linear" is reserved for future use.

    Returns
    -------
    soft_target : np.ndarray
        Float32 array, shape (H, W), values in [0, 1].
        Original foreground pixels are guaranteed to be 1.0.
    """
    if mode not in ("cosine", "linear"):
        raise ValueError(f"Unknown soft target mode: '{mode}'. Use 'cosine' or 'linear'.")

    # Ensure binary mask is uint8 {0, 1}
    mask = binary_mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape

    # --- Step A: Protection dilation ---
    if protection_radius > 0:
        # Ellipse structuring element: diameter = 2*radius + 1
        kernel_size = 2 * protection_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        protected_fg = cv2.dilate(mask, kernel, iterations=1)
    else:
        protected_fg = mask.copy()

    # --- Step B: Outward distance transform ---
    # We need: distance from each background pixel to nearest protected FG pixel.
    # cv2.distanceTransform computes distance of each NON-ZERO pixel to nearest
    # ZERO pixel.  So we invert: background=1 (non-zero), foreground=0 (zero).
    # Then the distance of each background pixel is its distance to nearest FG.
    bg_mask = 1 - protected_fg  # background=1, protected_fg=0
    distance = cv2.distanceTransform(
        bg_mask.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
    )
    # distance: float32, shape (H, W)
    # distance[protected_fg == 1] == 0
    # distance[background] > 0

    # --- Step C: Feather ---
    if transition_width <= 0:
        # No transition: binary mask
        soft_target = protected_fg.astype(np.float32)
    elif mode == "cosine":
        soft_target = _cosine_feather(protected_fg, distance, transition_width)
    elif mode == "linear":
        soft_target = _linear_feather(protected_fg, distance, transition_width)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Guarantee: original foreground must be exactly 1.0
    soft_target[mask == 1] = 1.0

    # Clamp for safety
    np.clip(soft_target, 0.0, 1.0, out=soft_target)

    return soft_target


def _cosine_feather(
    protected_fg: np.ndarray,
    distance: np.ndarray,
    transition_width: int,
) -> np.ndarray:
    """Cosine feather: M(d) = 0.5 * (1 + cos(π * d / T)).

    Properties:
        M(0) = 1  (at protected foreground boundary)
        M(T) = 0  (at transition_width distance)
    """
    T = float(transition_width)
    soft = np.zeros_like(distance, dtype=np.float32)

    # Protected foreground → M = 1
    soft[protected_fg == 1] = 1.0

    # Transition zone: 0 < d < T
    transition_mask = (protected_fg == 0) & (distance < T) & (distance > 0)
    d = distance[transition_mask]
    soft[transition_mask] = 0.5 * (1.0 + np.cos(np.pi * d / T))

    # Far background (d >= T): M = 0  (already initialized to 0)

    return soft


def _linear_feather(
    protected_fg: np.ndarray,
    distance: np.ndarray,
    transition_width: int,
) -> np.ndarray:
    """Linear feather: M(d) = max(0, 1 - d / T).

    Reserved for future ablation.
    """
    T = float(transition_width)
    soft = np.zeros_like(distance, dtype=np.float32)

    # Protected foreground → M = 1
    soft[protected_fg == 1] = 1.0

    # Transition zone: 0 < d < T
    transition_mask = (protected_fg == 0) & (distance < T) & (distance > 0)
    d = distance[transition_mask]
    soft[transition_mask] = 1.0 - d / T

    return soft
