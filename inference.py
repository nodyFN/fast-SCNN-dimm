#!/usr/bin/env python3
"""
Inference script for Fast-SCNN Dimming.

Processes single images or entire folders and outputs:
- Soft mask grayscale
- Binary threshold visualization
- Heatmap
- Dimmed preview
- Side-by-side comparison

Usage
-----
::

    # Single image
    python inference.py --weights checkpoints/.../best_val_loss.pt --input photo.jpg

    # Folder
    python inference.py --weights checkpoints/.../best_val_loss.pt --input images/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from config import Config
from models.fast_scnn_dimming import FastSCNNDimming
from utils.checkpoint import load_checkpoint
from utils.visualization import (
    create_dimmed_preview,
    mask_to_heatmap,
    protection_to_brightness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for Fast-SCNN Dimming")
    p.add_argument("--weights", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--input", type=str, required=True, help="Input image or folder")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--height", type=int, default=None,
                   help="Input height (default: auto from checkpoint or 128)")
    p.add_argument("--width", type=int, default=None,
                   help="Input width (default: auto from checkpoint or 224)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--min-brightness", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary threshold for mask visualization")
    p.add_argument("--resize-to-original", action="store_true",
                   help="Resize predictions back to original image resolution")
    return p.parse_args()


def find_images(path: Path) -> List[Path]:
    """Find all image files in a path (single file or directory)."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        else:
            raise ValueError(f"Not a supported image file: {path}")
    elif path.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(path.glob(f"*{ext}"))
            images.extend(path.glob(f"*{ext.upper()}"))
        return sorted(set(images))
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def preprocess(
    image_bgr: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    """Preprocess image for model input.

    Returns
    -------
    tensor : [1, 3, H, W], float32, normalized
    """
    # Resize
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    # BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # [H, W, C] → [C, H, W]
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)


def run_inference(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
) -> np.ndarray:
    """Run model inference on a single image.

    Returns
    -------
    prob : ndarray [H, W], float32 in [0, 1]
        At model resolution (height × width).
    """
    tensor = preprocess(image_bgr, height, width).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
    return prob[0, 0].cpu().numpy()


def save_outputs(
    output_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    prob: np.ndarray,
    min_brightness: float,
    threshold: float,
    resize_to_original: bool,
) -> None:
    """Save all inference outputs for a single image."""
    orig_h, orig_w = image_bgr.shape[:2]
    model_h, model_w = prob.shape

    # Optionally resize prediction back to original resolution
    if resize_to_original and (orig_h != model_h or orig_w != model_w):
        prob_display = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        display_image = image_bgr
    else:
        prob_display = prob
        display_image = cv2.resize(image_bgr, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

    h, w = prob_display.shape

    # 1. Soft mask grayscale
    soft_gray = (np.clip(prob_display, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{stem}_soft_mask.png"), soft_gray)

    # 2. Binary threshold visualization
    binary_vis = ((prob_display >= threshold).astype(np.uint8)) * 255
    cv2.imwrite(str(output_dir / f"{stem}_binary.png"), binary_vis)

    # 3. Heatmap
    heatmap = mask_to_heatmap(prob_display)
    cv2.imwrite(str(output_dir / f"{stem}_heatmap.png"), heatmap)

    # 4. Dimmed preview
    dimmed = create_dimmed_preview(display_image, prob_display, min_brightness)
    cv2.imwrite(str(output_dir / f"{stem}_dimmed.jpg"), dimmed)

    # 5. Side-by-side comparison
    panel_orig = display_image.copy()
    panel_binary = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
    panel_heatmap = heatmap
    brightness_map = mask_to_heatmap(
        protection_to_brightness(prob_display, min_brightness)
    )
    panel_dimmed = dimmed

    side_by_side = np.concatenate(
        [panel_orig, panel_binary, panel_heatmap, brightness_map, panel_dimmed],
        axis=1,
    )
    cv2.imwrite(str(output_dir / f"{stem}_comparison.jpg"), side_by_side)


def main() -> None:
    args = parse_args()
    cfg = Config()

    if args.device:
        cfg.device = args.device
    device = cfg.resolve_device()
    logger.info(f"Device: {device}")

    # Load model
    model = FastSCNNDimming(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        dropout_p=cfg.dropout_p,
    ).to(device)
    ckpt = load_checkpoint(args.weights, model, map_location=device, weights_only=True)
    model.eval()
    logger.info(f"Loaded weights from: {args.weights}")

    # Auto-resolve inference resolution from checkpoint or fallback
    saved_config = ckpt.get("config", {})
    infer_h = args.height if args.height is not None else int(
        saved_config.get("val_height", saved_config.get("train_height", 128))
    )
    infer_w = args.width if args.width is not None else int(
        saved_config.get("val_width", saved_config.get("train_width", 224))
    )
    logger.info(f"Resolution for inference: H={infer_h}, W={infer_w}")

    # Find images
    input_path = Path(args.input)
    images = find_images(input_path)
    logger.info(f"Found {len(images)} images")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = cfg.inference_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process
    for img_path in images:
        logger.info(f"Processing: {img_path.name}")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logger.warning(f"Failed to load: {img_path}")
            continue

        prob = run_inference(model, image_bgr, infer_h, infer_w, device)

        save_outputs(
            output_dir=output_dir,
            stem=img_path.stem,
            image_bgr=image_bgr,
            prob=prob,
            min_brightness=args.min_brightness,
            threshold=args.threshold,
            resize_to_original=args.resize_to_original,
        )

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
