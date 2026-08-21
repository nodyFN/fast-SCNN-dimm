#!/bin/bash

# ==============================================================================
# test.sh - Batch Inference and Testing Script
#
# Available parameters for inference.py:
#
# [Model Weights]
#   --weights: Path to the trained model checkpoint (.pt)
#
# [Input & Output]
#   --input: Path to a single image file or a directory of images
#   --output-dir: Output directory to save predictions (visualizations, heatmaps)
#
# [Resolution Settings]
#   --height / --width: Inference spatial resolution (resizes input to this size)
#   --resize-to-original: Rescales the final output mask back to the original image shape
#
# [Device Selection]
#   --device: Target hardware device ('cuda' | 'cpu')
# ==============================================================================

set -x

MODEL_TIMESTAMP="20260818_130508"
CHECKPOINT_TYPE="latest.pt"
HEIGHT="512"
WIDTH="1024"

INPUT="../dataset/data/test/images"
OUTPUT_DIR="test_result/real/2head_finetune_3/test"

rm -rf ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 pipenv run python inference.py \
  --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} \
  --input ${INPUT} \
  --output-dir ${OUTPUT_DIR} \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --resize-to-original
