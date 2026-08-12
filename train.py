#!/usr/bin/env python3
"""
Training script for Fast-SCNN Dimming: Foreground Protection Soft Mask Prediction.

Supports:
- CLI override of all config parameters
- AMP training
- TensorBoard logging
- Checkpoint save/resume
- Validation with full metrics
- Training visualization
- Smoke test with synthetic data
- Periodic checkpoint saving

Usage
-----
::

    # Full training
    python train.py --data-root duts_data

    # Custom resolution and batch size
    python train.py --data-root duts_data --train-height 128 --train-width 224 \\
                    --batch-size 16 --epochs 200

    # Smoke test (no real data needed)
    python train.py --smoke-test

    # Resume from checkpoint
    python train.py --data-root duts_data --resume checkpoints/.../latest.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset import build_dataloaders
from models.fast_scnn_dimming import FastSCNNDimming, count_parameters
from utils.checkpoint import load_checkpoint, load_pretrained_weights, save_checkpoint
from utils.losses import DimmingLoss, build_criterion
from utils.metrics import MetricAccumulator, format_metrics
from utils.scheduler import build_scheduler
from utils.seed import seed_everything
from utils.visualization import plot_training_curves, save_training_visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Argument parser
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with config override."""
    p = argparse.ArgumentParser(
        description="Train Fast-SCNN Dimming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data-root", type=str, default="duts_data")

    # Resolution
    p.add_argument("--train-height", type=int, default=None)
    p.add_argument("--train-width", type=int, default=None)
    p.add_argument("--val-height", type=int, default=None)
    p.add_argument("--val-width", type=int, default=None)

    # Training
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-tqdm", action="store_true")

    # Optimizer
    p.add_argument("--optimizer", type=str, default=None)
    p.add_argument("--learning-rate", "--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)

    # Scheduler
    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--poly-power", type=float, default=None)

    # Loss weights
    p.add_argument("--lambda-bce", type=float, default=None)
    p.add_argument("--lambda-l1", type=float, default=None)
    p.add_argument("--lambda-protect", type=float, default=None)

    # Soft target
    p.add_argument("--protection-radius", type=int, default=None)
    p.add_argument("--transition-width", type=int, default=None)
    p.add_argument("--soft-target-mode", type=str, default=None)

    # Model
    p.add_argument("--num-classes", type=int, default=None,
                   help="Number of output classes (default: 1 for binary dimming, >1 for multiclass pretraining)")
    p.add_argument("--dropout-p", type=float, default=None)

    # Checkpoint & Transfer Learning
    p.add_argument("--pretrained", type=str, default=None,
                   help="Path to pretrained weights to initialize backbone (ignores classifier head shape mismatch)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume full training from")
    p.add_argument("--checkpoint-save-interval", type=int, default=None)

    # Mask
    p.add_argument("--allow-threshold", action="store_true")

    # Visualization
    p.add_argument("--num-vis-samples", type=int, default=None)
    p.add_argument("--vis-interval", type=int, default=None)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Run quick smoke test with synthetic data")

    return p.parse_args()


def apply_args_to_config(args: argparse.Namespace, cfg: Config) -> Config:
    """Override config defaults with CLI arguments."""
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.train_height is not None:
        cfg.train_height = args.train_height
    if args.train_width is not None:
        cfg.train_width = args.train_width
    if args.val_height is not None:
        cfg.val_height = args.val_height
    if args.val_width is not None:
        cfg.val_width = args.val_width
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.no_amp:
        cfg.amp = False
    if args.no_tqdm:
        cfg.no_tqdm = True
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.scheduler is not None:
        cfg.scheduler = args.scheduler
    if args.poly_power is not None:
        cfg.poly_power = args.poly_power
    if args.lambda_bce is not None:
        cfg.lambda_bce = args.lambda_bce
    if args.lambda_l1 is not None:
        cfg.lambda_l1 = args.lambda_l1
    if args.lambda_protect is not None:
        cfg.lambda_protect = args.lambda_protect
    if args.protection_radius is not None:
        cfg.protection_radius = args.protection_radius
    if args.transition_width is not None:
        cfg.transition_width = args.transition_width
    if args.soft_target_mode is not None:
        cfg.soft_target_mode = args.soft_target_mode
    if args.dropout_p is not None:
        cfg.dropout_p = args.dropout_p
    if args.num_classes is not None:
        cfg.num_classes = args.num_classes
    if args.pretrained is not None:
        cfg.pretrained = args.pretrained
    if args.resume is not None:
        cfg.resume = args.resume
    if args.checkpoint_save_interval is not None:
        cfg.checkpoint_save_interval = args.checkpoint_save_interval
    if args.allow_threshold:
        cfg.allow_threshold = True
    if args.num_vis_samples is not None:
        cfg.num_vis_samples = args.num_vis_samples
    if args.vis_interval is not None:
        cfg.vis_interval = args.vis_interval
    return cfg


# ===========================================================================
# Smoke test
# ===========================================================================


def run_smoke_test(cfg: Config) -> None:
    """Smoke test: verify entire pipeline with synthetic data.

    Tests:
    1. Model forward
    2. Soft target generation
    3. Loss computation
    4. Backward pass
    5. Optimizer step
    6. Validation metrics
    7. Checkpoint save/load
    """
    from utils.soft_target import generate_soft_target

    logger.info("=" * 60)
    logger.info("SMOKE TEST — synthetic data, no real dataset required")
    logger.info("=" * 60)

    device = cfg.resolve_device()
    logger.info(f"Device: {device}")

    # 1. Model
    logger.info("\n1. Building model...")
    model = FastSCNNDimming(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        dropout_p=cfg.dropout_p,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"   Total params: {total_params:,}")
    logger.info(f"   Trainable:    {trainable_params:,}")

    # 2. Forward
    logger.info("\n2. Model forward pass...")
    B = 2
    x = torch.randn(B, 3, cfg.train_height, cfg.train_width, device=device)
    model.train()
    logits = model(x)
    assert logits.shape == (B, 1, cfg.train_height, cfg.train_width), (
        f"Expected [B, 1, {cfg.train_height}, {cfg.train_width}], got {list(logits.shape)}"
    )
    logger.info(f"   logits shape: {list(logits.shape)} ✓")

    # 3. Soft target generation
    logger.info("\n3. Soft target generation...")
    binary_np = np.zeros((cfg.train_height, cfg.train_width), dtype=np.uint8)
    binary_np[40:80, 80:160] = 1  # synthetic foreground rectangle
    soft_np = generate_soft_target(
        binary_np,
        protection_radius=cfg.protection_radius,
        transition_width=cfg.transition_width,
        mode=cfg.soft_target_mode,
    )
    assert soft_np.shape == (cfg.train_height, cfg.train_width)
    assert soft_np.min() >= 0.0 and soft_np.max() <= 1.0
    assert np.all(soft_np[binary_np == 1] == 1.0)
    logger.info(f"   soft target range: [{soft_np.min():.3f}, {soft_np.max():.3f}] ✓")
    logger.info(f"   foreground all 1.0: ✓")

    # 4. Loss
    logger.info("\n4. Loss computation...")
    criterion = DimmingLoss(
        lambda_bce=cfg.lambda_bce,
        lambda_l1=cfg.lambda_l1,
        lambda_protect=cfg.lambda_protect,
    )
    soft_target = torch.rand(B, 1, cfg.train_height, cfg.train_width, device=device)
    binary_mask = (soft_target > 0.5).float()
    losses = criterion(logits, soft_target, binary_mask)
    logger.info(f"   total: {losses['total'].item():.4f}")
    logger.info(f"   bce:   {losses['bce'].item():.4f}")
    logger.info(f"   l1:    {losses['l1'].item():.4f}")
    logger.info(f"   protect: {losses['protect'].item():.4f}")

    # 5. Backward
    logger.info("\n5. Backward pass...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    logger.info("   backward + optimizer step ✓")

    # 6. Validation metrics
    logger.info("\n6. Validation metrics...")
    model.eval()
    with torch.no_grad():
        logits_val = model(x)
        prob = torch.sigmoid(logits_val)

    metrics = MetricAccumulator()
    metrics.update(prob, soft_target, binary_mask)
    results = metrics.compute()
    for name, value in results.items():
        logger.info(f"   {name}: {value:.4f}")

    # 7. Checkpoint save/load and resume verification
    logger.info("\n7. Checkpoint save/load & resume verification...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "latest.pt"
        scheduler = build_scheduler(
            cfg.scheduler, optimizer,
            total_iters=100, epochs=cfg.epochs,
            poly_power=cfg.poly_power,
        )
        # Epoch 0: save initial best
        save_checkpoint(
            ckpt_path,
            epoch=0,
            global_step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=0.5,
            best_fg_protection=0.90,
            best_soft_mae=0.05,
            config=asdict(cfg),
            seed=cfg.seed,
        )
        # Epoch 1: update best metrics and save latest.pt
        save_checkpoint(
            ckpt_path,
            epoch=1,
            global_step=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=0.4,
            best_fg_protection=0.95,
            best_soft_mae=0.03,
            config=asdict(cfg),
            seed=cfg.seed,
        )
        assert ckpt_path.exists()

        # Load and verify latest.pt contains the updated best metrics
        model2 = FastSCNNDimming(
            ppm_pool_sizes=cfg.ppm_pool_sizes,
            dropout_p=cfg.dropout_p,
        ).to(device)
        ckpt = load_checkpoint(ckpt_path, model2)
        assert ckpt["epoch"] == 1
        assert ckpt["global_step"] == 2
        assert abs(ckpt["best_val_loss"] - 0.4) < 1e-6, f"Expected 0.4, got {ckpt['best_val_loss']}"
        assert abs(ckpt["best_fg_protection"] - 0.95) < 1e-6
        assert abs(ckpt["best_soft_mae"] - 0.03) < 1e-6
        logger.info(
            f"   loaded latest.pt: epoch={ckpt['epoch']}, step={ckpt['global_step']}, "
            f"best_val_loss={ckpt['best_val_loss']} ✓"
        )

    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST PASSED ✓")
    logger.info("=" * 60)


# ===========================================================================
# Training / validation loops
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    cfg: Config,
    global_step: int,
    writer: SummaryWriter,
) -> tuple[float, int]:
    """Train for one epoch.

    Returns
    -------
    avg_loss : float
    global_step : int (updated)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    try:
        from tqdm import tqdm
        use_tqdm = not cfg.no_tqdm
    except ImportError:
        use_tqdm = False

    iterator = loader
    if use_tqdm:
        iterator = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        soft_targets = batch["soft_mask"].to(device, non_blocking=True)
        binary_masks = batch["binary_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # AMP forward
        use_amp = cfg.amp and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            losses = criterion(logits, soft_targets, binary_masks)

        # Backward
        scaler.scale(losses["total"]).backward()

        if cfg.gradient_clip_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.gradient_clip_max_norm
            )

        scaler.step(optimizer)
        scaler.update()

        # Step scheduler (PolyLR is per-iteration)
        if cfg.scheduler == "poly":
            scheduler.step()

        # Logging
        batch_loss = losses["total"].item()
        total_loss += batch_loss
        num_batches += 1
        global_step += 1

        if global_step % 50 == 0:
            writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
            writer.add_scalar("train/loss_bce", losses["bce"].item(), global_step)
            writer.add_scalar("train/loss_l1", losses["l1"].item(), global_step)
            writer.add_scalar("train/loss_protect", losses["protect"].item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if use_tqdm:
            iterator.set_postfix(
                loss=f"{batch_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    # Step cosine scheduler (per-epoch)
    if cfg.scheduler == "cosine":
        scheduler.step()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> tuple[float, dict]:
    """Run validation.

    Returns
    -------
    avg_loss : float
    metrics : dict
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metric_acc = MetricAccumulator()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        soft_targets = batch["soft_mask"].to(device, non_blocking=True)
        binary_masks = batch["binary_mask"].to(device, non_blocking=True)

        use_amp = cfg.amp and device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            losses = criterion(logits, soft_targets, binary_masks)

        total_loss += losses["total"].item()
        num_batches += 1

        prob = torch.sigmoid(logits)
        metric_acc.update(prob, soft_targets, binary_masks)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = metric_acc.compute()
    return avg_loss, metrics


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    args = parse_args()
    cfg = Config()
    cfg = apply_args_to_config(args, cfg)

    # Smoke test
    if args.smoke_test:
        seed_everything(cfg.seed, cfg.deterministic)
        run_smoke_test(cfg)
        return

    # Reproducibility
    seed_everything(cfg.seed, cfg.deterministic)

    # Device
    if args.device:
        cfg.device = args.device
    device = cfg.resolve_device()
    logger.info(f"Device: {device}")

    # Ensure directories
    cfg.ensure_dirs()

    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.checkpoint_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = cfg.training_image_dir / timestamp
    vis_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tb_dir = cfg.tensorboard_dir / timestamp
    writer = SummaryWriter(log_dir=str(tb_dir))

    # Log config
    config_dict = asdict(cfg)
    # Convert Path objects to strings for JSON serialization
    config_json = {}
    for k, v in config_dict.items():
        config_json[k] = str(v) if isinstance(v, Path) else v
    logger.info(f"Config:\n{json.dumps(config_json, indent=2)}")

    # Save config to run directory
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    # Build model
    model = FastSCNNDimming(
        num_classes=cfg.num_classes,
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        dropout_p=cfg.dropout_p,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"Number of classes (output channels): {cfg.num_classes}")
    logger.info(f"Input resolution: H={cfg.train_height}, W={cfg.train_width}")
    logger.info(f"PyTorch input shape: [B, 3, {cfg.train_height}, {cfg.train_width}]")

    # Load pretrained backbone weights (e.g. from COCO-Stuff / ADE20K pretraining)
    if cfg.pretrained:
        logger.info(f"Loading pretrained backbone weights from: {cfg.pretrained}")
        load_pretrained_weights(cfg.pretrained, model, map_location=device)

    # Batch size guard
    if cfg.batch_size == 1:
        raise ValueError(
            "Batch size 1 is not supported by the current PPM BatchNorm configuration. "
            "Please use batch_size >= 2."
        )

    # Build dataloaders
    loaders = build_dataloaders(
        data_root=cfg.data_root,
        train_height=cfg.train_height,
        train_width=cfg.train_width,
        val_height=cfg.val_height,
        val_width=cfg.val_width,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        protection_radius=cfg.protection_radius,
        transition_width=cfg.transition_width,
        soft_target_mode=cfg.soft_target_mode,
        allow_threshold=cfg.allow_threshold,
        aug_brightness_limit=cfg.aug_brightness_limit,
        aug_contrast_limit=cfg.aug_contrast_limit,
        aug_hflip_p=cfg.aug_hflip_p,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples:   {len(val_loader.dataset)}")

    # Loss criterion
    criterion = build_criterion(
        num_classes=cfg.num_classes,
        lambda_bce=cfg.lambda_bce,
        lambda_l1=cfg.lambda_l1,
        lambda_protect=cfg.lambda_protect,
    )
    if cfg.num_classes > 1:
        logger.info(
            f"Loss criterion: MulticlassCrossEntropyLoss (num_classes={cfg.num_classes}, ignore_index=255)"
        )
    else:
        logger.info(
            f"Loss weights: BCE={cfg.lambda_bce}, L1={cfg.lambda_l1}, "
            f"Protect={cfg.lambda_protect}"
        )

    # Optimizer
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    logger.info(f"Optimizer: {cfg.optimizer}, LR={cfg.learning_rate}")

    # Scheduler
    total_iters = len(train_loader) * cfg.epochs
    scheduler = build_scheduler(
        cfg.scheduler,
        optimizer,
        total_iters=total_iters,
        epochs=cfg.epochs,
        poly_power=cfg.poly_power,
        cosine_eta_min=cfg.cosine_eta_min,
    )
    logger.info(f"Scheduler: {cfg.scheduler}")

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_fg_protection = 0.0
    best_soft_mae = float("inf")

    if cfg.resume:
        logger.info(f"Resuming from: {cfg.resume}")
        ckpt = load_checkpoint(
            cfg.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_fg_protection = ckpt.get("best_fg_protection", 0.0)
        best_soft_mae = ckpt.get("best_soft_mae", float("inf"))
        logger.info(
            f"Resumed: epoch={start_epoch}, step={global_step}, "
            f"best_val_loss={best_val_loss:.4f}"
        )

    # Training info
    logger.info(f"\nSeed: {cfg.seed}")
    logger.info(f"Protection radius: {cfg.protection_radius}")
    logger.info(f"Transition width: {cfg.transition_width}")
    logger.info(f"Soft target mode: {cfg.soft_target_mode}")
    logger.info(f"Epochs: {cfg.epochs}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"AMP: {cfg.amp}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("")

    # Training history for plotting curves
    history: dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "fg_iou": [],
        "miou": [],
        "dice": [],
        "fg_mean_protection": [],
        "soft_mae": [],
        "far_bg_leakage": [],
    }

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # Train
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, cfg, global_step, writer,
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, cfg,
        )

        elapsed = time.time() - t0

        # Log
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"fg_prot={val_metrics['fg_mean_protection']:.4f} | "
            f"soft_mae={val_metrics['soft_mae']:.4f} | "
            f"fg_iou={val_metrics['fg_iou']:.4f} | "
            f"miou={val_metrics['miou']:.4f} | "
            f"dice={val_metrics['dice']:.4f} | "
            f"bg_leak={val_metrics['far_bg_leakage']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        # TensorBoard
        writer.add_scalar("val/loss", val_loss, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)

        # Record history & plot curves
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["fg_iou"].append(val_metrics.get("fg_iou", 0.0))
        history["miou"].append(val_metrics.get("miou", 0.0))
        history["dice"].append(val_metrics.get("dice", 0.0))
        history["fg_mean_protection"].append(val_metrics.get("fg_mean_protection", 0.0))
        history["soft_mae"].append(val_metrics.get("soft_mae", 0.0))
        history["far_bg_leakage"].append(val_metrics.get("far_bg_leakage", 0.0))

        plot_training_curves(history, vis_dir / "training_curves.png")

        # Save checkpoint helper
        def _save(tag: str) -> None:
            save_checkpoint(
                run_dir / f"{tag}.pt",
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
                best_fg_protection=best_fg_protection,
                best_soft_mae=best_soft_mae,
                val_metrics=val_metrics,
                config=config_json,
                seed=cfg.seed,
            )

        # Step 1: Update best metrics and set flags
        is_best_val = False
        is_best_fg = False
        is_best_soft = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best_val = True

        fg_prot = val_metrics["fg_mean_protection"]
        if fg_prot > best_fg_protection:
            best_fg_protection = fg_prot
            is_best_fg = True

        soft_mae = val_metrics["soft_mae"]
        if soft_mae < best_soft_mae:
            best_soft_mae = soft_mae
            is_best_soft = True

        # Step 2: Save latest checkpoint (always includes latest best metrics)
        _save("latest")

        # Step 3: Save individual best checkpoints when a new best is achieved
        if is_best_val:
            _save("best_val_loss")
            logger.info(f"  → New best val_loss: {best_val_loss:.4f}")

        if is_best_fg:
            _save("best_fg_protection")
            logger.info(f"  → New best fg_protection: {best_fg_protection:.4f}")

        if is_best_soft:
            _save("best_soft_mae")
            logger.info(f"  → New best soft_mae: {best_soft_mae:.4f}")

        # Periodic checkpoint
        if (
            cfg.checkpoint_save_interval > 0
            and (epoch + 1) % cfg.checkpoint_save_interval == 0
        ):
            _save(f"epoch_{epoch:04d}")

        # Visualization
        if epoch % cfg.vis_interval == 0 and cfg.num_vis_samples > 0:
            model.eval()
            vis_images_list, vis_masks_list, vis_soft_list, vis_pred_list = [], [], [], []
            total_collected = 0
            with torch.no_grad():
                for vis_batch in val_loader:
                    imgs = vis_batch["image"].to(device)
                    b_masks = vis_batch["binary_mask"]
                    s_masks = vis_batch["soft_mask"]
                    preds = torch.sigmoid(model(imgs)).cpu()

                    vis_images_list.append(vis_batch["image"])
                    vis_masks_list.append(b_masks)
                    vis_soft_list.append(s_masks)
                    vis_pred_list.append(preds)

                    total_collected += imgs.size(0)
                    if total_collected >= cfg.num_vis_samples:
                        break

            if vis_images_list:
                all_vis_imgs = torch.cat(vis_images_list, dim=0)[:cfg.num_vis_samples]
                all_vis_masks = torch.cat(vis_masks_list, dim=0)[:cfg.num_vis_samples]
                all_vis_soft = torch.cat(vis_soft_list, dim=0)[:cfg.num_vis_samples]
                all_vis_preds = torch.cat(vis_pred_list, dim=0)[:cfg.num_vis_samples]

                save_training_visualization(
                    vis_dir / f"epoch_{epoch:04d}.jpg",
                    images=all_vis_imgs,
                    binary_masks=all_vis_masks,
                    soft_targets=all_vis_soft,
                    predictions=all_vis_preds,
                    num_samples=cfg.num_vis_samples,
                    min_brightness=cfg.min_brightness,
                )

    writer.close()

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best val_loss:       {best_val_loss:.4f}")
    logger.info(f"Best fg_protection:  {best_fg_protection:.4f}")
    logger.info(f"Best soft_mae:       {best_soft_mae:.4f}")
    logger.info(f"Checkpoints: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
