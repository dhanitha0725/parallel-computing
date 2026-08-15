#include <mpi.h>
#include "common.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string dataset_path = "data/susy_0.5percent.csv";
    size_t sample_size = 25000;
    int k = DEFAULT_K;

    if (argc >= 2) dataset_path = argv[1];
    if (argc >= 3) sample_size = std::stoull(argv[2]);
    if (argc >= 4) k = std::stoi(argv[3]);

    size_t n_train = 0;
    size_t n_test = 0;
    std::vector<double> flat_train_feats;
    std::vector<int> train_labels;
    std::vector<double> flat_test_feats;
    std::vector<int> actual_test_labels;
    std::vector<int> seq_predictions;
    std::vector<int> par_predictions;

    double ts_ms = 0.0;

    std::vector<int> send_counts_feat(num_procs, 0);
    std::vector<int> displs_feat(num_procs, 0);
    std::vector<int> recv_counts_pred(num_procs, 0);
    std::vector<int> displs_pred(num_procs, 0);

    // Rank 0: Load data, split, and compute sequential baseline
    if (rank == 0) {
        std::cout << "\n=======================================================\n";
        std::cout << "    High-Performance Computing Benchmark (OpenMPI)     \n";
        std::cout << "=======================================================\n";
        std::cout << "MPI Processes (p) : " << num_procs << "\n";
        std::cout << "Dataset Path      : " << dataset_path << "\n";
        std::cout << "Target Samples    : " << sample_size << "\n";
        std::cout << "K Value           : " << k << "\n\n";

        std::cout << "[Step 1] Loading dataset..." << std::flush;
        std::vector<Event> full_dataset;
        if (!load_susy_csv(dataset_path, full_dataset, sample_size)) {
            std::cerr << "\nError: Failed to load dataset from " << dataset_path << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        std::cout << " Done (" << full_dataset.size() << " samples).\n";

        std::cout << "[Step 2] Partitioning (80/20) & Normalizing..." << std::flush;
        std::vector<Event> train_events, test_events;
        train_test_split(full_dataset, train_events, test_events, 0.80);
        standardize_dataset(train_events, test_events);

        n_train = train_events.size();
        n_test = test_events.size();

        flat_train_feats.resize(n_train * NUM_FEATURES);
        train_labels.resize(n_train);
        for (size_t i = 0; i < n_train; ++i) {
            train_labels[i] = train_events[i].label;
            for (int j = 0; j < NUM_FEATURES; ++j) {
                flat_train_feats[i * NUM_FEATURES + j] = train_events[i].features[j];
            }
        }

        flat_test_feats.resize(n_test * NUM_FEATURES);
        actual_test_labels.resize(n_test);
        for (size_t i = 0; i < n_test; ++i) {
            actual_test_labels[i] = test_events[i].label;
            for (int j = 0; j < NUM_FEATURES; ++j) {
                flat_test_feats[i * NUM_FEATURES + j] = test_events[i].features[j];
            }
        }

        seq_predictions.resize(n_test, 0);
        par_predictions.resize(n_test, 0);

        // Partition test samples across MPI ranks
        int current_disp = 0;
        for (int p = 0; p < num_procs; ++p) {
            int chunk = static_cast<int>(n_test / num_procs) + (p < static_cast<int>(n_test % num_procs) ? 1 : 0);
            recv_counts_pred[p] = chunk;
            displs_pred[p] = current_disp;
            send_counts_feat[p] = chunk * NUM_FEATURES;
            displs_feat[p] = current_disp * NUM_FEATURES;
            current_disp += chunk;
        }
        std::cout << " Done.\n";

        // Compute Sequential Baseline (Ts) on Rank 0
        std::cout << "[Step 3] Running Sequential Baseline (Ts)..." << std::flush;
        auto seq_t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n_test; ++i) {
            const double* feat = &flat_test_feats[i * NUM_FEATURES];
            seq_predictions[i] = knn_predict_single(feat, flat_train_feats.data(), train_labels.data(), n_train, k);
        }
        auto seq_t1 = std::chrono::high_resolution_clock::now();
        ts_ms = std::chrono::duration<double, std::milli>(seq_t1 - seq_t0).count();
        std::cout << " Done (Ts = " << std::fixed << std::setprecision(2) << ts_ms << " ms).\n";
    }

    // Step 4: Broadcast metadata and training set
    int meta[2];
    if (rank == 0) {
        meta[0] = static_cast<int>(n_train);
        meta[1] = static_cast<int>(n_test);
    }
    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);
    n_train = meta[0];
    n_test = meta[1];

    if (rank != 0) {
        flat_train_feats.resize(n_train * NUM_FEATURES);
        train_labels.resize(n_train);
    }

    MPI_Bcast(flat_train_feats.data(), static_cast<int>(n_train * NUM_FEATURES), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(train_labels.data(), static_cast<int>(n_train), MPI_INT, 0, MPI_COMM_WORLD);

    int local_n_test = static_cast<int>(n_test / num_procs) + (rank < static_cast<int>(n_test % num_procs) ? 1 : 0);
    std::vector<double> local_test_feats(local_n_test * NUM_FEATURES);
    std::vector<int> local_predictions(local_n_test, 0);

    MPI_Scatterv(rank == 0 ? flat_test_feats.data() : nullptr,
                 send_counts_feat.data(),
                 displs_feat.data(),
                 MPI_DOUBLE,
                 local_test_feats.data(),
                 local_n_test * NUM_FEATURES,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[Step 5] Running Parallel OpenMPI Execution (Tp across " << num_procs << " ranks)..." << std::flush;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double par_t0 = MPI_Wtime();

    for (int i = 0; i < local_n_test; ++i) {
        const double* feat = &local_test_feats[i * NUM_FEATURES];
        local_predictions[i] = knn_predict_single(feat, flat_train_feats.data(), train_labels.data(), n_train, k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double par_t1 = MPI_Wtime();

    double local_tp_ms = (par_t1 - par_t0) * 1000.0;
    double max_tp_ms = 0.0;

    MPI_Reduce(&local_tp_ms, &max_tp_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_predictions.data(),
                local_n_test,
                MPI_INT,
                rank == 0 ? par_predictions.data() : nullptr,
                recv_counts_pred.data(),
                displs_pred.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << " Done (Tp = " << std::fixed << std::setprecision(2) << max_tp_ms << " ms).\n";

        // Verify exact correctness between sequential and parallel predictions
        int mismatches = 0;
        for (size_t i = 0; i < n_test; ++i) {
            if (seq_predictions[i] != par_predictions[i]) mismatches++;
        }

        ClassificationMetrics seq_metrics = evaluate_predictions(actual_test_labels, seq_predictions);
        ClassificationMetrics par_metrics = evaluate_predictions(actual_test_labels, par_predictions);

        double speedup = (max_tp_ms > 0.0) ? (ts_ms / max_tp_ms) : 0.0;
        double efficiency = (num_procs > 0) ? (speedup / num_procs) * 100.0 : 0.0;

        std::cout << "\n=========================================================================\n";
        std::cout << "                 PARALLEL COMPUTING METRICS SUMMARY                      \n";
        std::cout << "=========================================================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Number of Processes (p)       : " << num_procs << "\n";
        std::cout << "  Sequential Time (Ts)          : " << ts_ms << " ms (" << (ts_ms / 1000.0) << " s)\n";
        std::cout << "  Parallel Time (Tp)            : " << max_tp_ms << " ms (" << (max_tp_ms / 1000.0) << " s)\n";
        std::cout << "  Speedup (Sp = Ts / Tp)        : " << std::setprecision(4) << speedup << " x\n";
        std::cout << "  Parallel Efficiency (Ep)      : " << std::setprecision(2) << efficiency << " %\n";
        std::cout << "  Model Accuracy                : " << (par_metrics.accuracy() * 100.0) << " %\n";
        std::cout << "  Prediction Parity Verification: " << (mismatches == 0 ? "PASSED (100% Match)" : "FAILED") << "\n";
        std::cout << "=========================================================================\n";

        // Append to benchmark CSV
        std::ofstream csv("benchmark_summary.csv", std::ios::app);
        if (csv.is_open()) {
            csv << num_procs << "," << ts_ms << "," << max_tp_ms << "," << speedup << "," << efficiency << "," << par_metrics.accuracy() * 100.0 << "\n";
            csv.close();
        }
    }

    MPI_Finalize();
    return 0;
}
