# Parallel and High Performance Computing: Parallel Classification on SUSY Dataset

This repository contains the complete implementation, benchmarking suite, and performance evaluation for the **Parallel Classification on the SUSY Dataset** group assignment using **C++** and **OpenMP (Multi-threading)**.

---

## 1. Project Overview & Objectives

* **Problem**: Binary classification of particle-collision events into **Class 1 (SUSY Signal)** vs. **Class 0 (Background)** based on 18 numerical physical features.
* **Dataset**: Monte Carlo simulated SUSY dataset. In accordance with the assignment guidelines, the **0.5% common subset (~25,000 collision events)** is utilized (20,000 train / 5,000 test).
* **Algorithm**: **K-Nearest Neighbors (KNN)** with Euclidean Distance ($L_2$ metric) and standard Z-score feature normalization.
* **Parallel Framework**: **OpenMP** (Shared-memory multi-threading model).
* **Target Hardware**: Intel® Core™ i7-13700HX (Testing with 1, 2, 4, 8 threads).

---

## 2. High-Performance Computing (HPC) Architecture & OpenMP Strategy

### Computational Bottleneck
With $N_{train} = 20,000$ training samples and $N_{test} = 5,000$ test queries across $D = 18$ feature dimensions:
$$\text{Total Pairwise Operations} = 5,000 \times 20,000 \times 18 = 1.8 \times 10^8 \text{ floating-point operations}$$
Each query instance's nearest neighbors can be computed independently, making the prediction phase **embarrassingly parallel**.

### OpenMP Multi-threading Implementation
1. **Shared Memory Access**: The training dataset matrices (`flat_train_feats`, `train_labels`) reside in shared RAM and are read concurrently by all threads without data replication overhead.
2. **Loop Parallelization**: The outer query loop over the 5,000 test events is distributed among threads using:
   ```cpp
   #pragma omp parallel for schedule(dynamic, 64) num_threads(num_threads)
   for (size_t i = 0; i < n_test; ++i) {
       predictions[i] = knn_predict_single(...);
   }
   ```
3. **Lock-Free / Thread-Safe Execution**: Each thread writes its prediction to a disjoint index `predictions[i]`. There are zero mutex locks, zero race conditions, and zero false sharing in the critical path.
4. **Execution Timing**: Precise execution wall-clock time is measured using `omp_get_wtime()`.

---

## 3. Mathematical Formulations

* **Sequential Execution Time ($T_s$)**: Measured execution time of the computationally intensive stage using 1 thread.
* **Parallel Execution Time ($T_p$)**: Measured execution time using $p$ threads.
* **Speedup ($S_p$)**:
  $$S_p = \frac{T_s}{T_p}$$
* **Parallel Efficiency ($E_p$)**:
  $$E_p = \frac{S_p}{p} \times 100\% = \frac{T_s}{p \times T_p} \times 100\%$$

---

## 4. Empirical Benchmark Results (Intel® Core™ i7-13700HX)

| Implementation | Threads | Time | Speedup | Efficiency |
|:---|:---:|:---:|:---:|:---:|
| **Sequential** | 1 | **423 ms** | **1.00x** | **100%** |
| **OpenMP** | 2 | **286 ms** | **1.48x** | **73.9%** |
| **OpenMP** | 4 | **123 ms** | **3.44x** | **86.1%** |
| **OpenMP** | 8 | **72 ms** | **5.91x** | **73.9%** |

* **Classification Accuracy**: **75.46 %**
* **Prediction Parity**: **100% Exact Match (0 mismatches across all thread counts)**

---

## 5. How to Compile & Run

### Automated Execution (Single Command via Windows PowerShell):
```powershell
./scripts/run_all.ps1
```

### Manual Execution (Linux / WSL):
```bash
# 1. Compile binaries
./scripts/build.sh

# 2. Run full benchmark suite
./scripts/run_all.sh
```
