"""
Checkpoint save / load utilities.

Features
--------
- Atomic write (write to tmp, then rename) to reduce corruption risk.
- Stores: epoch, global_step, model, optimizer, scheduler, scaler,
  best metrics, config, seed.
- Handles ``module.`` prefix from DataParallel / DDP.
- Two modes: full resume vs. weights-only.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path | str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any] = None,
    best_val_loss: float = float("inf"),
    best_fg_protection: float = 0.0,
    best_soft_mae: float = float("inf"),
    val_metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
    seed: int = 42,
) -> None:
    """Save a training checkpoint with atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_loss": best_val_loss,
        "best_fg_protection": best_fg_protection,
        "best_soft_mae": best_soft_mae,
        "val_metrics": val_metrics or {},
        "config": config or {},
        "seed": seed,
    }

    # Atomic write: save to temp file first, then rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp"
    )
    os.close(tmp_fd)

    try:
        torch.save(state, tmp_path)
        Path(tmp_path).replace(path)
        logger.info(f"Checkpoint saved to {path}")
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[str | torch.device] = None,
    weights_only: bool = False,
    filter_shape_mismatch: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint with automatic shape-mismatch filtering.

    Parameters
    ----------
    path : Path or str
        Path to checkpoint file.
    model : nn.Module
        Target PyTorch model.
    optimizer, scheduler, scaler : optional
        State containers to restore if weights_only is False.
    map_location : str or torch.device, optional
        Device mapping.
    weights_only : bool
        If True, only load model weights (ignore optimizer, scheduler, epoch, etc.).
    filter_shape_mismatch : bool
        If True, skip layers whose tensor shapes don't match the current model (e.g.
        when transferring a backbone pretrained on COCO-Stuff/ADE20K to a binary
        segmentation model with a different classifier head).

    Returns
    -------
    dict with all checkpoint data (epoch, global_step, best metrics, config, …)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Handle direct state_dict vs full checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format in {path}")

    # Handle DataParallel / DDP module. prefix
    state_dict = _strip_module_prefix(state_dict)
    model_state = model.state_dict()

    matched_dict = {}
    mismatched_keys = []
    skipped_keys = []

    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                matched_dict[k] = v
            elif filter_shape_mismatch:
                mismatched_keys.append((k, list(v.shape), list(model_state[k].shape)))
            else:
                matched_dict[k] = v
        else:
            skipped_keys.append(k)

    missing_keys = set(model_state.keys()) - set(matched_dict.keys())

    # Load matching weights
    model.load_state_dict(matched_dict, strict=False)

    logger.info(
        f"Loaded {len(matched_dict)}/{len(model_state)} matching parameter tensors from {path}"
    )

    if mismatched_keys:
        logger.info(
            f"Shape-mismatched layers skipped ({len(mismatched_keys)}): "
            + ", ".join(
                f"'{k}' (ckpt: {ckpt_s} -> model: {model_s})"
                for k, ckpt_s, model_s in mismatched_keys
            )
            + " — these layers will remain initialized for the current task."
        )

    if missing_keys - {k for k, _, _ in mismatched_keys}:
        unmatched_missing = missing_keys - {k for k, _, _ in mismatched_keys}
        logger.info(f"Missing keys in checkpoint (initialized randomly): {unmatched_missing}")

    if not weights_only and isinstance(checkpoint, dict):
        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")

        if scheduler and checkpoint.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")

        if scaler and checkpoint.get("scaler_state_dict"):
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load scaler state: {e}")

    return checkpoint


def load_pretrained_weights(
    path: Path | str,
    model: nn.Module,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:
    """Load pretrained backbone weights into model, automatically skipping mismatched heads.

    Starts training fresh from epoch 0.
    """
    return load_checkpoint(
        path=path,
        model=model,
        weights_only=True,
        filter_shape_mismatch=True,
        map_location=map_location,
    )


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'module.' prefix added by DataParallel / DDP."""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned
