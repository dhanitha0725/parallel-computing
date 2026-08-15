#!/bin/bash
set -e

# Change directory to project root
cd "$(dirname "$0")/.."

mkdir -p bin
mkdir -p data

echo "======================================================="
echo "   Compiling Parallel Classification C++ (OpenMP)      "
echo "======================================================="

# 1. Compile dataset extractor utility
echo "[1/4] Compiling prepare_subset..."
g++ -O3 -std=c++17 src/prepare_subset.cpp -o bin/prepare_subset

# 2. Compile Sequential KNN
echo "[2/4] Compiling sequential_knn..."
g++ -O3 -march=native -std=c++17 src/sequential_knn.cpp -o bin/sequential_knn

# 3. Compile Parallel OpenMP KNN
echo "[3/4] Compiling parallel_knn_omp..."
g++ -O3 -march=native -fopenmp -std=c++17 src/parallel_knn_omp.cpp -o bin/parallel_knn_omp

# 4. Compile Benchmark OpenMP Runner
echo "[4/4] Compiling benchmark_omp..."
g++ -O3 -march=native -fopenmp -std=c++17 src/benchmark_omp.cpp -o bin/benchmark_omp

echo "======================================================="
echo "   Build completed successfully in ./bin/              "
echo "======================================================="
