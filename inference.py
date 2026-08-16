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
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from config import Config
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.visualization import (
    add_panel_label,
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
    p = argparse.ArgumentParser(description="Inference for Fast-SCNN Dimming / Dual-Head")
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
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run model inference on a single image.

    Returns
    -------
    fine_prob : ndarray [H, W], float32 in [0, 1]
        At model resolution (height × width).
    coarse_prob : ndarray [H, W], float32 in [0, 1] or None
        At model resolution (height × width).
    """
    tensor = preprocess(image_bgr, height, width).to(device)
    with torch.inference_mode():
        out = model(tensor)
        coarse_prob = None
        if isinstance(out, dict):
            prob = out.get("fine_prob", torch.sigmoid(out["fine_logits"]))
            coarse_prob = out.get("coarse_prob", torch.sigmoid(out["coarse_logits"]))
        else:
            prob = torch.sigmoid(out)

    fine_prob_np = prob[0, 0].cpu().numpy()
    coarse_prob_np = coarse_prob[0, 0].cpu().numpy() if coarse_prob is not None else None
    return fine_prob_np, coarse_prob_np


def save_outputs(
    output_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    prob: np.ndarray,
    coarse_prob: Optional[np.ndarray],
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
        coarse_prob_display = cv2.resize(coarse_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) if coarse_prob is not None else None
        display_image = image_bgr
    else:
        prob_display = prob
        coarse_prob_display = coarse_prob
        display_image = cv2.resize(image_bgr, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

    # 1. Soft mask grayscale (0~255)
    soft_gray = (np.clip(prob_display, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{stem}_soft_gray.png"), soft_gray)
    cv2.imwrite(str(output_dir / f"{stem}_soft_mask.png"), soft_gray)

    # 2. Coarse mask grayscale (Option 1 & 2 verification)
    coarse_gray = None
    if coarse_prob_display is not None:
        coarse_gray = (np.clip(coarse_prob_display, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{stem}_coarse_gray.png"), coarse_gray)

    # 3. Binary threshold visualization
    binary_vis = ((prob_display >= threshold).astype(np.uint8)) * 255
    cv2.imwrite(str(output_dir / f"{stem}_binary.png"), binary_vis)

    # 4. Heatmap
    heatmap = mask_to_heatmap(prob_display)
    cv2.imwrite(str(output_dir / f"{stem}_heatmap.png"), heatmap)

    # 5. Dimmed preview
    dimmed = create_dimmed_preview(display_image, prob_display, min_brightness)
    cv2.imwrite(str(output_dir / f"{stem}_dimmed.jpg"), dimmed)

    # 6. Side-by-side comparison (with labeled headers)
    panel_orig = add_panel_label(display_image.copy(), "Original RGB")
    panel_binary = add_panel_label(cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR), f"Binary (>={threshold})")
    
    panels = [panel_orig, panel_binary]
    if coarse_gray is not None:
        panel_coarse_gray = add_panel_label(cv2.cvtColor(coarse_gray, cv2.COLOR_GRAY2BGR), "Coarse Mask (Gray)")
        panels.append(panel_coarse_gray)

    panel_gray = add_panel_label(cv2.cvtColor(soft_gray, cv2.COLOR_GRAY2BGR), "Soft Mask (Gray)")
    panel_heatmap = add_panel_label(heatmap, "Soft Mask (Heatmap)")
    brightness_map = add_panel_label(
        mask_to_heatmap(protection_to_brightness(prob_display, min_brightness)),
        "Brightness Map",
    )
    panel_dimmed = add_panel_label(dimmed, "Dimmed Preview")

    panels.extend([panel_gray, panel_heatmap, brightness_map, panel_dimmed])

    side_by_side = np.concatenate(panels, axis=1)
    cv2.imwrite(str(output_dir / f"{stem}_comparison.jpg"), side_by_side)


def main() -> None:
    args = parse_args()
    cfg = Config()

    if args.device:
        cfg.device = args.device
    device = cfg.resolve_device()
    # Inspect checkpoint to auto-resolve num_classes and resolution
    raw_ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    saved_config = raw_ckpt.get("config", {}) if isinstance(raw_ckpt, dict) else {}
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt

    num_classes = 1
    if isinstance(state_dict, dict) and "classifier.conv.bias" in state_dict:
        num_classes = state_dict["classifier.conv.bias"].shape[0]
    elif "num_classes" in saved_config:
        num_classes = int(saved_config["num_classes"])

    model_name = "fast_scnn_dimming"
    if isinstance(state_dict, dict):
        if any("coarse_head" in k for k in state_dict.keys()):
            model_name = "fast_scnn_dual_head"
        elif "model_name" in saved_config:
            model_name = saved_config["model_name"]

    logger.info(f"Model output channels (num_classes): {num_classes}")
    logger.info(f"Detected model architecture: {model_name}")

    refinement_head = saved_config.get("refinement_head", "multiscale")
    if isinstance(state_dict, dict) and model_name == "fast_scnn_dual_head":
        if any("refinement_head.dsconv" in k for k in state_dict.keys()):
            refinement_head = "legacy_h8"
        elif any("refinement_head.h8_proj" in k for k in state_dict.keys()):
            refinement_head = "multiscale"
        logger.info(f"Detected refinement head type: {refinement_head}")

    # Load model
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        dropout_p=cfg.dropout_p,
        refinement_head=refinement_head,
        prompt_gate_mode=saved_config.get("prompt_gate_mode", "bidirectional"),
        prompt_gate_strength=saved_config.get("prompt_gate_strength", 0.5),
    ).to(device)
    ckpt = load_checkpoint(args.weights, model, map_location=device, weights_only=True)
    model.eval()
    logger.info(f"Loaded weights from: {args.weights}")

    # Auto-resolve inference resolution from checkpoint or fallback
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

        prob, coarse_prob = run_inference(model, image_bgr, infer_h, infer_w, device)

        save_outputs(
            output_dir=output_dir,
            stem=img_path.stem,
            image_bgr=image_bgr,
            prob=prob,
            coarse_prob=coarse_prob,
            min_brightness=args.min_brightness,
            threshold=args.threshold,
            resize_to_original=args.resize_to_original,
        )

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
