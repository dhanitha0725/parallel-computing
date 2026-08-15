#!/bin/bash
set -e

# Change directory to project root
cd "$(dirname "$0")/.."

DATASET="data/susy_0.5percent.csv"
FULL_DATASET="data/supersymmetry_dataset.csv"

# Step 1: Ensure dataset subset exists
if [ ! -f "$DATASET" ]; then
    if [ -f "$FULL_DATASET" ]; then
        echo "[!] Extracting 25,000 records (0.5% subset) from $FULL_DATASET..."
        ./bin/prepare_subset "$FULL_DATASET" "$DATASET" 25000
    else
        echo "[ERROR] Cannot find $FULL_DATASET or $DATASET in ./data/"
        exit 1
    fi
fi

# Step 2: Execute OpenMP Benchmark Suite
./bin/benchmark_omp "$DATASET" 25000 5
