#!/bin/bash

# ==============================================================================
# eval.sh - Model Evaluation Script
#
# Available parameters for evaluate.py:
#
# [Model Weights]
#   --weights: Path to the trained model checkpoint (.pt)
#
# [Dataset Split]
#   --data-root: Path to the dataset root containing images and masks folders
#   --split: Target split to evaluate ('val' | 'test' | 'train')
#
# [Evaluation Output]
#   --output-dir: Directory to save the final JSON metrics summary
#
# [Resolution & Batch Settings]
#   --val-height / --val-width: Spatial resolution to resize inputs (e.g., 512 1024)
#   --batch-size: Batch size for the evaluation loader
#
# [Threshold & Mask Settings]
#   --threshold: Threshold value to binarize predicted masks for binary metrics (e.g., 0.9)
#   --allow-threshold: Automatically thresholds GT masks that are not strictly binary
#
# [Device Selection]
#   --device: Target hardware device ('cuda' | 'cpu')
# ==============================================================================

set -x

MODEL_TIMESTAMP="20260818_130508"
CHECKPOINT_TYPE="latest.pt"
HEIGHT="512"
WIDTH="1024"
BATCH_SIZE="8"
THRESHOLD="0.9"

# 1. Evaluate on Real dataset (Test split)
DATA_ROOT="../dataset/data"
SPLIT="test"
OUTPUT_DIR="eval_result/real/2head_finetune_3/${SPLIT}"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=1 pipenv run python evaluate.py \
  --allow-threshold \
  --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} \
  --data-root ${DATA_ROOT} \
  --split ${SPLIT} \
  --output-dir ${OUTPUT_DIR} \
  --val-height ${HEIGHT} \
  --val-width ${WIDTH} \
  --batch-size ${BATCH_SIZE} \
  --threshold ${THRESHOLD}

# 2. Evaluate on Real dataset (Val split)
DATA_ROOT="../dataset/data"
SPLIT="val"
OUTPUT_DIR="eval_result/real/2head_finetune_3/${SPLIT}"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=1 pipenv run python evaluate.py \
  --allow-threshold \
  --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} \
  --data-root ${DATA_ROOT} \
  --split ${SPLIT} \
  --output-dir ${OUTPUT_DIR} \
  --val-height ${HEIGHT} \
  --val-width ${WIDTH} \
  --batch-size ${BATCH_SIZE} \
  --threshold ${THRESHOLD}

# 3. Evaluate on DUTS dataset (Test split)
DATA_ROOT="duts_data"
SPLIT="test"
OUTPUT_DIR="eval_result/real/2head_finetune_3/duts_${SPLIT}"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=1 pipenv run python evaluate.py \
  --allow-threshold \
  --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} \
  --data-root ${DATA_ROOT} \
  --split ${SPLIT} \
  --output-dir ${OUTPUT_DIR} \
  --val-height ${HEIGHT} \
  --val-width ${WIDTH} \
  --batch-size ${BATCH_SIZE} \
  --threshold ${THRESHOLD}
