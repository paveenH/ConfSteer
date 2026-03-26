#!/usr/bin/env bash
set -euo pipefail

MODEL="llama3"
LAYER=25
PCA=50
RATIO=1.0
SAMPLES="samples/${MODEL}/samples_binary_all.npz"

echo "=============================="
echo "  ConfSteer Binary Classifier"
echo "  Model  : ${MODEL}"
echo "  Layer  : ${LAYER}"
echo "  PCA    : ${PCA}"
echo "  Ratio  : ${RATIO}"
echo "  Samples: ${SAMPLES}"
echo "=============================="

python classifier_binary.py \
    --model   "${MODEL}" \
    --layer   "${LAYER}" \
    --pca     "${PCA}" \
    --ratio   "${RATIO}" \
    --samples "${SAMPLES}" \
    --visualize
