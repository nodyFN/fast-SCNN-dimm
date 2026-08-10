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
) -> Dict[str, Any]:
    """Load a checkpoint.

    Parameters
    ----------
    weights_only : bool
        If True, only load model weights (ignore optimizer, scheduler, etc.).

    Returns
    -------
    dict with all checkpoint data (epoch, global_step, best metrics, …)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Handle DataParallel / DDP module. prefix
    state_dict = checkpoint.get("model_state_dict", {})
    state_dict = _strip_module_prefix(state_dict)
    model_state = model.state_dict()

    # Check for missing / unexpected keys
    missing = set(model_state.keys()) - set(state_dict.keys())
    unexpected = set(state_dict.keys()) - set(model_state.keys())
    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

    model.load_state_dict(state_dict, strict=False)

    if not weights_only:
        if optimizer and "optimizer_state_dict" in checkpoint:
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


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'module.' prefix added by DataParallel / DDP."""
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned
