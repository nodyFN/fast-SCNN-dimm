#!/usr/bin/env python3
"""
Evaluation script for Fast-SCNN Dimming.

Usage
-----
::

    python evaluate.py --weights checkpoints/.../best_val_loss.pt --data-root duts_data
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from config import Config
from dataset import build_dataloaders
from models import build_model, count_parameters
from utils.checkpoint import load_checkpoint
from utils.losses import build_criterion
from utils.metrics import MetricAccumulator, format_metrics
from utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Fast-SCNN Dimming / Dual-Head")
    p.add_argument("--weights", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--data-root", type=str, default="duts_data")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--val-height", type=int, default=None)
    p.add_argument("--val-width", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--protection-radius", type=int, default=None)
    p.add_argument("--transition-width", type=int, default=None)
    p.add_argument("--allow-threshold", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    # Override config
    cfg.data_root = Path(args.data_root)
    if args.val_height is not None:
        cfg.val_height = args.val_height
    if args.val_width is not None:
        cfg.val_width = args.val_width
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    if args.device:
        cfg.device = args.device
    if args.protection_radius is not None:
        cfg.protection_radius = args.protection_radius
    if args.transition_width is not None:
        cfg.transition_width = args.transition_width
    if args.allow_threshold:
        cfg.allow_threshold = True

    device = cfg.resolve_device()
    seed_everything(cfg.seed)
    logger.info(f"Device: {device}")

    # Inspect checkpoint to auto-resolve model type and num_classes
    raw_ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    saved_config = raw_ckpt.get("config", {}) if isinstance(raw_ckpt, dict) else {}
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt

    num_classes = 1
    if isinstance(state_dict, dict) and "classifier.conv.bias" in state_dict:
        num_classes = state_dict["classifier.conv.bias"].shape[0]
    elif "num_classes" in saved_config:
        num_classes = int(saved_config["num_classes"])

    model_name = "fast_scnn_dimming"
    is_dual_head = False
    if isinstance(state_dict, dict):
        if any("coarse_head" in k for k in state_dict.keys()):
            model_name = "fast_scnn_dual_head"
            is_dual_head = True
        elif "model_name" in saved_config:
            model_name = saved_config["model_name"]
            is_dual_head = "dual" in model_name.lower()

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
    logger.info(f"Loaded weights from: {args.weights}")
    if "epoch" in ckpt:
        logger.info(f"  Checkpoint epoch: {ckpt['epoch']}")

    # Try to restore config from checkpoint
    if saved_config:
        if args.val_height is None and "val_height" in saved_config:
            cfg.val_height = int(saved_config["val_height"])
        elif args.val_height is None and "train_height" in saved_config:
            cfg.val_height = int(saved_config["train_height"])

        if args.val_width is None and "val_width" in saved_config:
            cfg.val_width = int(saved_config["val_width"])
        elif args.val_width is None and "train_width" in saved_config:
            cfg.val_width = int(saved_config["train_width"])

        if args.protection_radius is None and "protection_radius" in saved_config:
            cfg.protection_radius = int(saved_config["protection_radius"])
        if args.transition_width is None and "transition_width" in saved_config:
            cfg.transition_width = int(saved_config["transition_width"])
        if "soft_target_mode" in saved_config:
            cfg.soft_target_mode = str(saved_config["soft_target_mode"])

    logger.info(f"Resolution for evaluation: H={cfg.val_height}, W={cfg.val_width}")
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,}")

    # Build dataloader
    loaders = build_dataloaders(
        data_root=cfg.data_root,
        train_height=cfg.val_height,
        train_width=cfg.val_width,
        val_height=cfg.val_height,
        val_width=cfg.val_width,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=False,
        protection_radius=cfg.protection_radius,
        transition_width=cfg.transition_width,
        soft_target_mode=cfg.soft_target_mode,
        allow_threshold=cfg.allow_threshold,
    )

    loader = loaders.get(args.split)
    if loader is None:
        raise FileNotFoundError(f"Split '{args.split}' not found in {cfg.data_root}")
    logger.info(f"Evaluating on {args.split} split: {len(loader.dataset)} samples")

    # Evaluate
    model.eval()
    criterion = build_criterion(
        num_classes=num_classes,
        lambda_bce=cfg.lambda_bce,
        lambda_l1=cfg.lambda_l1,
        lambda_protect=cfg.lambda_protect,
        is_dual_head=is_dual_head,
        lambda_coarse=saved_config.get("lambda_coarse", 0.5),
    )
    metric_acc = MetricAccumulator()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            soft_targets = batch["soft_mask"].to(device, non_blocking=True)
            binary_masks = batch["binary_mask"].to(device, non_blocking=True)

            outputs = model(images)
            losses = criterion(outputs, soft_targets, binary_masks)

            total_loss += losses["total"].item()
            num_batches += 1

            if isinstance(outputs, dict):
                main_logits = outputs["fine_logits"]
                prob = outputs.get("fine_prob", torch.sigmoid(main_logits))
            else:
                prob = torch.sigmoid(outputs)
            metric_acc.update(prob, soft_targets, binary_masks)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = metric_acc.compute()
    metrics["val_loss"] = avg_loss

    # Print results
    logger.info(f"\nAverage loss: {avg_loss:.4f}")
    logger.info(format_metrics(metrics))

    # Save JSON summary
    output_dir = Path(args.output_dir) if args.output_dir else cfg.evaluation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"eval_{args.split}_{timestamp}.json"

    summary = {
        "weights": str(args.weights),
        "split": args.split,
        "num_samples": len(loader.dataset),
        "val_height": cfg.val_height,
        "val_width": cfg.val_width,
        "protection_radius": cfg.protection_radius,
        "transition_width": cfg.transition_width,
        "metrics": metrics,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nJSON summary saved to: {json_path}")


if __name__ == "__main__":
    main()
