#!/bin/bash
set -x

# MODEL_TIMESTAMP="20260817_091010"
MODEL_TIMESTAMP="20260818_130508"
CHECKPOINT_TYPE="latest.pt"
# HEIGHT="128"
# WIDTH="224"
# HEIGHT="256"
# WIDTH="256"
HEIGHT="512"
WIDTH="1024"
BATCH_SIZE="8"

THRESHOLD="0.9"

DATA_ROOT="../dataset/data"
SPLIT="test"
OUTPUT_DIR="eval_result/real/2head_finetune_3/${SPLIT}"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=1 pipenv run python evaluate.py --allow-threshold --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} --data-root ${DATA_ROOT} --split ${SPLIT} --output-dir ${OUTPUT_DIR} --val-height ${HEIGHT} --val-width ${WIDTH} --batch-size ${BATCH_SIZE} --threshold ${THRESHOLD}

DATA_ROOT="../dataset/data"
SPLIT="val"
OUTPUT_DIR="eval_result/real/2head_finetune_3/${SPLIT}"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=1 pipenv run python evaluate.py --allow-threshold --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} --data-root ${DATA_ROOT} --split ${SPLIT} --output-dir ${OUTPUT_DIR} --val-height ${HEIGHT} --val-width ${WIDTH} --batch-size ${BATCH_SIZE} --threshold ${THRESHOLD}


DATA_ROOT="duts_data"
SPLIT="test"
OUTPUT_DIR="eval_result/real/2head_finetune_3/duts_${SPLIT}"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=1 pipenv run python evaluate.py --allow-threshold --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} --data-root ${DATA_ROOT} --split ${SPLIT} --output-dir ${OUTPUT_DIR} --val-height ${HEIGHT} --val-width ${WIDTH} --batch-size ${BATCH_SIZE} --threshold ${THRESHOLD}

