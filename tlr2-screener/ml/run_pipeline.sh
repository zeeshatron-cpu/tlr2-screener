#!/bin/bash
# Run the full ML training pipeline
# Usage: cd ml && bash run_pipeline.sh

set -e
echo "=== TLR2 ML Pipeline ==="

echo "[1/4] Building curated dataset..."
python3 00_build_curated_dataset.py

echo "[2/4] Computing molecular features..."
python3 02_prepare_features.py

echo "[3/4] Training model..."
python3 03_train_model.py

echo "[4/4] Validating Pam3Cys-SNFKK..."
python3 04_validate_pam3cys.py

echo "Done. Model saved to model/tlr2_model.pkl"
