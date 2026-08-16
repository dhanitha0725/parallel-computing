#!/bin/bash
set -e

ensure_compiler() {
    if command -v g++ >/dev/null 2>&1; then
        return 0
    fi

    echo "[!] g++ not found. Installing C/C++ build tools..."

    if command -v pacman >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1; then
            sudo pacman -S --needed --noconfirm base-devel gcc
        else
            pacman -S --needed --noconfirm base-devel gcc
        fi
    elif command -v apt-get >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y build-essential
        else
            apt-get update
            apt-get install -y build-essential
        fi
    elif command -v dnf >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1; then
            sudo dnf groupinstall -y "Development Tools"
        else
            dnf groupinstall -y "Development Tools"
        fi
    elif command -v yum >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1; then
            sudo yum groupinstall -y "Development Tools"
        else
            yum groupinstall -y "Development Tools"
        fi
    elif command -v apk >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apk add --no-cache build-base
        else
            apk add --no-cache build-base
        fi
    else
        echo "[ERROR] No supported package manager found. Please install g++ manually."
        exit 1
    fi

    if ! command -v g++ >/dev/null 2>&1; then
        echo "[ERROR] g++ installation failed or is not on PATH."
        exit 1
    fi
}

# Change directory to project root
cd "$(dirname "$0")/.."

ensure_compiler

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
