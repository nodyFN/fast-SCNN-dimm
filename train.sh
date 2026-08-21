#!/bin/bash

# ==============================================================================
# train.sh - Model Training Script
#
# Available parameters for train.py:
#
# [Model Selection]
#   --model: Model architecture ('dimming' | 'dual_head')
#   --pretrained: Path to pretrained weights (.pt) to initialize backbone
#   --resume: Path to checkpoint (.pt) to resume training (restores optimizer/epoch)
#   --num-classes: Number of output channels (default: 1 for binary dimming)
#
# [Dataset & Input Resolution]
#   --data-root: Path to the dataset directory (containing train/val folders)
#   --train-height / --train-width: Resolution for training (e.g., 512 1024)
#   --val-height / --val-width: Resolution for validation (e.g., 512 1024)
#   --allow-threshold: Automatically thresholds masks that are not strictly binary
#
# [Hyperparameters & Training]
#   --batch-size: Batch size (e.g., 16)
#   --epochs: Total training epochs (e.g., 80)
#   --optimizer: Optimizer type ('adamw' | 'sgd')
#   --learning-rate / --lr: Initial learning rate (e.g., 1e-4)
#   --weight-decay: L2 regularization strength (e.g., 1e-4)
#   --scheduler: Learning rate scheduler ('poly' | 'cosine')
#   --poly-power: Power exponent for Poly LR scheduler
#   --seed: Random seed for reproducibility
#
# [Visualization & Checkpointing]
#   --vis-interval: Save visualization panels every N epochs (e.g., 1)
#   --num-vis-samples: Number of validation samples to visualize
#   --checkpoint-save-interval: Save checkpoints every N epochs (e.g., 1)
#   --no-tqdm: Disable progress bars for clean log files
#
# [Soft Target Formulation]
#   --protection-radius: Dilation radius for expanding foreground protection (e.g., 2)
#   --transition-width: Cosine feathering width for transition region (e.g., 4)
#
# [Dual-Head Optimization (Phase 1 & Phase 2)]
#   --coarse-only-epochs: Number of epochs to train ONLY CoarseHead in Phase 1 (e.g., 3)
#   --coarse-joint-training: If set, continues training CoarseHead in Phase 2
#   --coarse-edge-mask-kernel: Edge masking kernel size to ignore boundaries (default: 15)
#   --coarse-target-dilation-kernel: Dilation kernel size for coarse targets (default: 15)
# ==============================================================================

set -x

pipenv run python train.py \
  --model dual_head \
  --data-root ../dataset/data \
  --pretrained checkpoints/20260816_004441/latest.pt \
  --num-classes 1 \
  --train-height 512 \
  --train-width 1024 \
  --val-height 512 \
  --val-width 1024 \
  --batch-size 16 \
  --epochs 80 \
  --vis-interval 1 \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --scheduler poly \
  --poly-power 0.9 \
  --checkpoint-save-interval 1 \
  --seed 42 \
  --num-vis-samples 5 \
  --allow-threshold \
  --no-tqdm \
  --protection-radius 2 \
  --transition-width 4 \
  --coarse-only-epochs 3 \
  --coarse-joint-training
