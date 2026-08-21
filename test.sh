#!/bin/bash
set -x

MODEL_TIMESTAMP="20260818_130508"
CHECKPOINT_TYPE="latest.pt"
HEIGHT="512"
WIDTH="1024"
# HEIGHT="128"
# WIDTH="224"
# HEIGHT="1088"
# WIDTH="1920"

# INPUT="../dataset/data/val/images"
# OUTPUT_DIR="test_result/tem/val"
# rm -rf ${OUTPUT_DIR}
# CUDA_VISIBLE_DEVICES=0 pipenv run python inference.py --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} --input ${INPUT} --output-dir ${OUTPUT_DIR} --height ${HEIGHT} --width ${WIDTH} --resize-to-original

INPUT="../dataset/data/test/images"
# INPUT="duts_data/demo"
OUTPUT_DIR="test_result/real/2head_finetune_3/test"
rm -rf ${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=0 pipenv run python inference.py --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} --input ${INPUT} --output-dir ${OUTPUT_DIR} --height ${HEIGHT} --width ${WIDTH} --resize-to-original

# INPUT="duts_data/val/images"
# OUTPUT_DIR="test_result/dimm_duts/test"
# rm -rf ${OUTPUT_DIR}
# CUDA_VISIBLE_DEVICES=0 pipenv run python inference.py --weights checkpoints/${MODEL_TIMESTAMP}/${CHECKPOINT_TYPE} --input ${INPUT} --output-dir ${OUTPUT_DIR} --height ${HEIGHT} --width ${WIDTH} --resize-to-original

