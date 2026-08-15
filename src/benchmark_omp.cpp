#include <omp.h>
#include "common.hpp"

struct BenchmarkResult {
    std::string implementation;
    int threads;
    double time_ms;
    double speedup;
    double efficiency_percent;
    double accuracy_percent;
};

int main(int argc, char* argv[]) {
    std::string dataset_path = "data/susy_0.5percent.csv";
    size_t sample_size = 25000;
    int k = DEFAULT_K;

    if (argc >= 2) dataset_path = argv[1];
    if (argc >= 3) sample_size = std::stoull(argv[2]);
    if (argc >= 4) k = std::stoi(argv[3]);

    std::cout << "=========================================================================\n";
    std::cout << "       HPC BENCHMARK: SEQUENTIAL vs. OPENMP MULTITHREADING (SUSY)        \n";
    std::cout << "=========================================================================\n";
    std::cout << "Dataset Path   : " << dataset_path << "\n";
    std::cout << "Target Samples : " << sample_size << " (0.5% SUSY dataset subset)\n";
    std::cout << "K Value        : " << k << "\n\n";

    // 1. Load Dataset
    std::cout << "[Step 1] Loading dataset..." << std::flush;
    std::vector<Event> full_dataset;
    if (!load_susy_csv(dataset_path, full_dataset, sample_size)) {
        std::cerr << "\nError: Could not load dataset from " << dataset_path << std::endl;
        return 1;
    }
    std::cout << " Done (" << full_dataset.size() << " samples loaded).\n";

    // 2. Train/Test Split & Standardization
    std::cout << "[Step 2] Splitting into 80% Train / 20% Test & Standardizing..." << std::flush;
    std::vector<Event> train_events, test_events;
    train_test_split(full_dataset, train_events, test_events, 0.80);
    standardize_dataset(train_events, test_events);

    size_t n_train = train_events.size();
    size_t n_test = test_events.size();

    std::vector<double> flat_train_feats(n_train * NUM_FEATURES);
    std::vector<int> train_labels(n_train);
    for (size_t i = 0; i < n_train; ++i) {
        train_labels[i] = train_events[i].label;
        for (int j = 0; j < NUM_FEATURES; ++j) {
            flat_train_feats[i * NUM_FEATURES + j] = train_events[i].features[j];
        }
    }

    std::vector<double> flat_test_feats(n_test * NUM_FEATURES);
    std::vector<int> actual_test_labels(n_test);
    for (size_t i = 0; i < n_test; ++i) {
        actual_test_labels[i] = test_events[i].label;
        for (int j = 0; j < NUM_FEATURES; ++j) {
            flat_test_feats[i * NUM_FEATURES + j] = test_events[i].features[j];
        }
    }
    std::cout << " Done (" << n_train << " train, " << n_test << " test).\n\n";

    std::vector<BenchmarkResult> results;
    std::vector<int> seq_predictions(n_test, 0);
    double ts_ms = 0.0;

    // 3. Run Sequential Baseline (1 Thread)
    std::cout << "[Step 3] Running Sequential Baseline (1 Thread)..." << std::flush;
    {
        double t0 = omp_get_wtime();
        for (size_t i = 0; i < n_test; ++i) {
            const double* cur_feat = &flat_test_feats[i * NUM_FEATURES];
            seq_predictions[i] = knn_predict_single(cur_feat, flat_train_feats.data(), train_labels.data(), n_train, k);
        }
        double t1 = omp_get_wtime();
        ts_ms = (t1 - t0) * 1000.0;
        
        ClassificationMetrics metrics = evaluate_predictions(actual_test_labels, seq_predictions);
        results.push_back({"Sequential", 1, ts_ms, 1.00, 100.0, metrics.accuracy() * 100.0});
        std::cout << " Done (Ts = " << std::fixed << std::setprecision(2) << ts_ms << " ms).\n";
    }

    // 4. Run OpenMP Benchmarks for Threads = 2, 4, 8
    std::vector<int> thread_counts = {2, 4, 8};
    for (int t : thread_counts) {
        std::cout << "[Step 4] Running OpenMP with " << t << " Threads..." << std::flush;
        std::vector<int> par_predictions(n_test, 0);

        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 64) num_threads(t)
        for (size_t i = 0; i < n_test; ++i) {
            const double* cur_feat = &flat_test_feats[i * NUM_FEATURES];
            par_predictions[i] = knn_predict_single(cur_feat, flat_train_feats.data(), train_labels.data(), n_train, k);
        }
        double t1 = omp_get_wtime();
        double tp_ms = (t1 - t0) * 1000.0;

        // Verify exact prediction parity
        int mismatches = 0;
        for (size_t i = 0; i < n_test; ++i) {
            if (seq_predictions[i] != par_predictions[i]) mismatches++;
        }

        ClassificationMetrics metrics = evaluate_predictions(actual_test_labels, par_predictions);
        double speedup = ts_ms / tp_ms;
        double efficiency = (speedup / t) * 100.0;

        results.push_back({"OpenMP", t, tp_ms, speedup, efficiency, metrics.accuracy() * 100.0});
        std::cout << " Done (Tp = " << std::fixed << std::setprecision(2) << tp_ms << " ms | Parity: " 
                  << (mismatches == 0 ? "100% MATCH" : "FAILED") << ").\n";
    }

    // 5. Display Formatted Results Table (Matching requested layout)
    std::cout << "\n=========================================================================\n";
    std::cout << "                          FINAL EXPERIMENTAL RESULTS                     \n";
    std::cout << "=========================================================================\n\n";

    printf("%-18s %10s %16s %14s %14s\n", "Implementation", "Threads", "Time", "Speedup", "Efficiency");
    std::cout << "-------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::string time_str = std::to_string(static_cast<int>(std::round(r.time_ms))) + " ms";
        char speedup_buf[32];
        snprintf(speedup_buf, sizeof(speedup_buf), "%.2fx", r.speedup);

        char eff_buf[32];
        if (r.implementation == "Sequential") {
            snprintf(eff_buf, sizeof(eff_buf), "100%%");
        } else {
            snprintf(eff_buf, sizeof(eff_buf), "%.1f%%", r.efficiency_percent);
        }

        printf("%-18s %10d %16s %14s %14s\n",
               r.implementation.c_str(),
               r.threads,
               time_str.c_str(),
               speedup_buf,
               eff_buf);
    }
    std::cout << "=========================================================================\n";

    // Write to CSV
    std::ofstream csv("benchmark_summary.csv");
    if (csv.is_open()) {
        csv << "Implementation,Threads,Time_ms,Speedup,Efficiency_percent,Accuracy_percent\n";
        for (const auto& r : results) {
            csv << r.implementation << "," << r.threads << "," << r.time_ms << "," 
                << r.speedup << "," << r.efficiency_percent << "," << r.accuracy_percent << "\n";
        }
        csv.close();
    }

    return 0;
}
