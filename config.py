"""
Centralized configuration for Fast-SCNN Dimming project.

This module defines all tunable parameters in one place.  CLI arguments in
train.py, evaluate.py, inference.py and export.py can override these defaults.

Terminology
-----------
- **Paper setting**: values explicitly stated in the Fast-SCNN paper.
- **Project decision**: values chosen for this project that are NOT specified
  by the paper (marked with [PROJECT DECISION] in comments).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root (the directory that contains this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    """All-in-one configuration container."""

    # ── Input resolution ─────────────────────────────────────────────
    # [PROJECT DECISION] W=224, H=128 for landscape TV image
    train_height: int = 128
    train_width: int = 224

    val_height: int = 128
    val_width: int = 224

    # ── Data paths ────────────────────────────────────────────────────
    data_root: Path = PROJECT_ROOT / "duts_data"

    # ── DataLoader ────────────────────────────────────────────────────
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # ── Mask handling ─────────────────────────────────────────────────
    # If True, grayscale values > 127 → 1, ≤ 127 → 0
    allow_threshold: bool = False

    # ── Soft target generation ────────────────────────────────────────
    # [PROJECT DECISION] Foreground-preserving dimming soft target
    soft_target_mode: str = "cosine"   # "cosine" | "linear"
    protection_radius: int = 2          # Dilation radius (ellipse kernel)
    transition_width: int = 8           # Cosine feather width in pixels

    # ── Model ─────────────────────────────────────────────────────────
    model_name: str = "fast_scnn_dimming"  # "fast_scnn_dimming" | "fast_scnn_dual_head"
    num_classes: int = 1               # Default 1 (binary dimming), >1 for multiclass pretraining
    # [PROJECT DECISION] PPM pool sizes — paper does not specify for Fast-SCNN
    ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)
    # [PROJECT DECISION] Dropout in classifier
    dropout_p: float = 0.1
    refinement_head: str = "multiscale" # "multiscale" | "legacy_h8" (for dual_head)
    prompt_gate_mode: str = "bidirectional"
    prompt_gate_strength: float = 0.5

    # ── Optimizer ─────────────────────────────────────────────────────
    # [PROJECT DECISION] AdamW (paper uses SGD)
    optimizer: str = "adamw"            # "sgd" | "adamw"
    learning_rate: float = 1e-3
    momentum: float = 0.9              # SGD only (paper: 0.9)
    weight_decay: float = 1e-4

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler: str = "poly"             # "poly" | "cosine"
    poly_power: float = 0.9            # Paper: 0.9
    cosine_eta_min: float = 1e-6

    # ── Training ──────────────────────────────────────────────────────
    epochs: int = 200
    amp: bool = True                    # Only effective when CUDA is available
    gradient_clip_max_norm: float = 1.0
    gradient_clip_enabled: bool = True
    freeze_bn: bool = False             # If True, freeze all BatchNorm parameters and running stats

    # ── Loss weights ──────────────────────────────────────────────────
    # [PROJECT DECISION] BCE + L1 + Foreground Protection (for num_classes == 1)
    lambda_bce: float = 1.0
    lambda_l1: float = 1.0
    lambda_protect: float = 2.0
    lambda_coarse: float = 0.5         # Weight for CoarseHead loss in DualHead model
    coarse_only_epochs: int = 5         # Number of epochs to train only CoarseHead in DualHead model
    coarse_edge_mask_kernel: int = 15   # Edge mask kernel size for CoarseHead loss (0 to disable)
    coarse_target_dilation_kernel: int = 15 # Target dilation kernel size for CoarseHead loss (0 to disable)

    # ── Checkpointing ─────────────────────────────────────────────────
    checkpoint_save_interval: int = 0   # 0 = disabled; N = save every N epochs

    # ── Directories ───────────────────────────────────────────────────
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    training_image_dir: Path = PROJECT_ROOT / "training_results"
    tensorboard_dir: Path = PROJECT_ROOT / "runs"
    export_dir: Path = PROJECT_ROOT / "exports"
    evaluation_dir: Path = PROJECT_ROOT / "evaluation_results"
    inference_dir: Path = PROJECT_ROOT / "inference_results"

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = 42
    deterministic: bool = False

    # ── Device ────────────────────────────────────────────────────────
    device: str = "auto"               # "auto" | "cuda" | "cpu"

    # ── Transfer learning & Checkpoint resume ─────────────────────────
    pretrained: Optional[str] = None   # Path to pretrained weights (e.g. COCO-Stuff / ADE20K backbone)
    resume: Optional[str] = None       # Path to checkpoint to resume full training from

    # ── ONNX export ───────────────────────────────────────────────────
    # [PROJECT DECISION] opset 17 — good PyTorch/TensorRT compatibility
    onnx_opset: int = 17

    # ── Visualization ─────────────────────────────────────────────────
    num_vis_samples: int = 4
    vis_interval: int = 1              # Save validation vis every N epochs
    min_brightness: float = 0.5        # For dimming simulation

    # ── Augmentation ──────────────────────────────────────────────────
    aug_brightness_limit: float = 0.2
    aug_contrast_limit: float = 0.2
    aug_hflip_p: float = 0.5
    aug_gamma_p: float = 0.5            # Random Gamma correction probability
    aug_color_jitter_p: float = 0.5     # Color Jitter probability
    aug_clahe_p: float = 0.3            # CLAHE contrast enhancement probability
    aug_hsv_p: float = 0.5              # Hue/Saturation/Value shift probability

    # ── Display ───────────────────────────────────────────────────────
    no_tqdm: bool = False

    def resolve_device(self) -> torch.device:
        """Return the torch.device to use."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in [
            self.checkpoint_dir,
            self.training_image_dir,
            self.tensorboard_dir,
            self.export_dir,
            self.evaluation_dir,
            self.inference_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
