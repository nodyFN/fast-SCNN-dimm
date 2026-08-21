#!/bin/bash

set -x

# pipenv run python train.py \
#   --model dimming \
#   --data-root duts_data \
#   --num-classes 1 \
#   --train-height 512 \
#   --train-width 1024 \
#   --val-height 512 \
#   --val-width 1024 \
#   --batch-size 16 \
#   --epochs 200 \
#   --vis-interval 1 \
#   --optimizer adamw \
#   --learning-rate 3e-4 \
#   --weight-decay 1e-4 \
#   --scheduler poly \
#   --poly-power 0.9 \
#   --checkpoint-save-interval 1 \
#   --seed 30 \
#   --num-vis-samples 5 \
#   --allow-threshold \
#   --no-tqdm \
#   --protection-radius 0 \
#   --transition-width 0 \
#   --aug-gamma-p 0.0 \
#   --aug-color-jitter-p 0.0 \
#   --aug-clahe-p 0.0 \
#   --aug-hsv-p 0.0




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
