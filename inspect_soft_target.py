#!/usr/bin/env python3
"""
Visual inspection tool for soft target generation parameters.

Outputs side-by-side comparisons of:
- Original RGB
- Binary GT
- Dilated protected region
- Distance map visualization
- Final soft target

Also supports parameter comparison grid for quick visual assessment.

Usage
-----
::

    # Basic usage with default parameters
    python inspect_soft_target.py --data-root duts_data --num-samples 10

    # Compare different parameters
    python inspect_soft_target.py --data-root duts_data --num-samples 5 --compare
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from utils.soft_target import generate_soft_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect soft target generation"
    )
    p.add_argument("--data-root", type=str, default="duts_data")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="soft_target_inspection")
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--protection-radius", type=int, default=2)
    p.add_argument("--transition-width", type=int, default=8)
    p.add_argument("--compare", action="store_true",
                   help="Generate parameter comparison grid")
    return p.parse_args()


def load_pairs(data_root: Path, split: str, max_n: int):
    """Load image/mask pairs."""
    img_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {data_root / split}")

    mask_map = {p.stem: p for p in mask_dir.glob("*.png")}
    pairs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem in mask_map:
            pairs.append((p, mask_map[p.stem]))
            if len(pairs) >= max_n:
                break
    return pairs


def visualize_sample(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    height: int,
    width: int,
    protection_radius: int,
    transition_width: int,
) -> np.ndarray:
    """Create a single-row visualization for one sample."""
    # Resize
    image_resized = cv2.resize(image_bgr, (width, height))
    mask_resized = cv2.resize(
        binary_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )
    mask_resized = (mask_resized > 0).astype(np.uint8)

    # Dilated region
    if protection_radius > 0:
        kernel_size = 2 * protection_radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(mask_resized, kernel, iterations=1)
    else:
        dilated = mask_resized.copy()

    # Distance map
    bg_mask = (1 - dilated).astype(np.uint8)
    distance = cv2.distanceTransform(
        bg_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    # Soft target
    soft = generate_soft_target(
        mask_resized,
        protection_radius=protection_radius,
        transition_width=transition_width,
    )

    # Build panels
    panel_image = image_resized

    panel_binary = cv2.cvtColor(
        (mask_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    panel_dilated = cv2.cvtColor(
        (dilated * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    # Highlight dilation-only region in green
    dilation_only = (dilated == 1) & (mask_resized == 0)
    panel_dilated[dilation_only] = [0, 255, 0]

    # Distance map visualization (normalize for visibility)
    max_dist = max(distance.max(), 1.0)
    dist_normalized = (distance / max_dist * 255).astype(np.uint8)
    panel_distance = cv2.applyColorMap(dist_normalized, cv2.COLORMAP_JET)

    # Soft target heatmap
    panel_soft = cv2.applyColorMap(
        (soft * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    return np.concatenate(
        [panel_image, panel_binary, panel_dilated, panel_distance, panel_soft],
        axis=1,
    )


def visualize_comparison(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Create parameter comparison grid for one sample.

    Rows: protection_radius = 0, 2, 4
    Columns: transition_width = 4, 8, 12
    """
    radii = [0, 2, 4]
    widths = [4, 8, 12]

    # Resize once
    image_resized = cv2.resize(image_bgr, (width, height))
    mask_resized = cv2.resize(
        binary_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )
    mask_resized = (mask_resized > 0).astype(np.uint8)

    rows = []
    for r in radii:
        panels = []
        for tw in widths:
            soft = generate_soft_target(
                mask_resized,
                protection_radius=r,
                transition_width=tw,
            )
            panel = cv2.applyColorMap(
                (soft * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            # Add text label
            label = f"r={r} tw={tw}"
            cv2.putText(
                panel, label, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))

    # Add original image as first column header
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(data_root, args.split, args.num_samples)
    logger.info(f"Loaded {len(pairs)} samples")

    for i, (img_path, mask_path) in enumerate(pairs):
        logger.info(f"Processing [{i+1}/{len(pairs)}]: {img_path.name}")

        image_bgr = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            logger.warning(f"Failed to load: {img_path.name}")
            continue

        # Convert mask
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Main visualization
        vis = visualize_sample(
            image_bgr, mask,
            args.height, args.width,
            args.protection_radius, args.transition_width,
        )
        cv2.imwrite(str(output_dir / f"{img_path.stem}_inspect.jpg"), vis)

        # Parameter comparison
        if args.compare:
            comp = visualize_comparison(
                image_bgr, mask,
                args.height, args.width,
            )
            cv2.imwrite(str(output_dir / f"{img_path.stem}_compare.jpg"), comp)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Panels (left to right): RGB | Binary GT | Dilated | Distance | Soft Target")


if __name__ == "__main__":
    main()
