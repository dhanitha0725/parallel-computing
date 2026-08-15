#include <mpi.h>
#include "common.hpp"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::string dataset_path = "data/supersymmetry_dataset.csv";
    size_t sample_size = 25000; // 0.5% common subset
    int k = DEFAULT_K;

    if (argc >= 2) {
        dataset_path = argv[1];
    }
    if (argc >= 3) {
        sample_size = std::stoull(argv[2]);
    }
    if (argc >= 4) {
        k = std::stoi(argv[3]);
    }

    if (rank == 0) {
        std::cout << "=======================================================\n";
        std::cout << "     Parallel K-Nearest Neighbors (KNN) with OpenMPI   \n";
        std::cout << "=======================================================\n";
        std::cout << "Processes (p) : " << num_procs << "\n";
        std::cout << "Dataset Path  : " << dataset_path << "\n";
        std::cout << "Sample Limit  : " << sample_size << " records (0.5% SUSY subset)\n";
        std::cout << "K Value       : " << k << "\n";
    }

    size_t n_train = 0;
    size_t n_test = 0;
    std::vector<double> flat_train_feats;
    std::vector<int> train_labels;
    std::vector<double> flat_test_feats;
    std::vector<int> actual_test_labels;
    std::vector<int> all_predictions;

    // Buffers for MPI_Scatterv and MPI_Gatherv
    std::vector<int> send_counts_feat(num_procs, 0);
    std::vector<int> displs_feat(num_procs, 0);
    std::vector<int> recv_counts_pred(num_procs, 0);
    std::vector<int> displs_pred(num_procs, 0);

    // Rank 0: Load data, split, and standardize
    if (rank == 0) {
        std::cout << "\n[1/4] Rank 0 loading dataset from disk..." << std::endl;
        std::vector<Event> full_dataset;
        if (!load_susy_csv(dataset_path, full_dataset, sample_size)) {
            std::cerr << "Rank 0: Failed to load dataset from " << dataset_path << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        std::cout << "Rank 0: Loaded " << full_dataset.size() << " records." << std::endl;

        std::cout << "[2/4] Partitioning into 80% Train / 20% Test and Standardizing..." << std::endl;
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

        all_predictions.resize(n_test, 0);

        // Compute chunk partitioning for each rank
        int current_disp = 0;
        for (int p = 0; p < num_procs; ++p) {
            int chunk_samples = static_cast<int>(n_test / num_procs) + (p < static_cast<int>(n_test % num_procs) ? 1 : 0);
            recv_counts_pred[p] = chunk_samples;
            displs_pred[p] = current_disp;

            send_counts_feat[p] = chunk_samples * NUM_FEATURES;
            displs_feat[p] = current_disp * NUM_FEATURES;

            current_disp += chunk_samples;
        }

        std::cout << "[3/4] Distributing data across " << num_procs << " MPI ranks..." << std::endl;
    }

    // Step 1: Broadcast dimensions (n_train and n_test) to all ranks
    int meta[2];
    if (rank == 0) {
        meta[0] = static_cast<int>(n_train);
        meta[1] = static_cast<int>(n_test);
    }
    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);
    n_train = meta[0];
    n_test = meta[1];

    // Allocate memory on worker ranks for training set
    if (rank != 0) {
        flat_train_feats.resize(n_train * NUM_FEATURES);
        train_labels.resize(n_train);
    }

    // Step 2: Broadcast the complete training set (features + labels) to all ranks
    MPI_Bcast(flat_train_feats.data(), static_cast<int>(n_train * NUM_FEATURES), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(train_labels.data(), static_cast<int>(n_train), MPI_INT, 0, MPI_COMM_WORLD);

    // Determine how many test samples this specific rank will process
    int local_n_test = static_cast<int>(n_test / num_procs) + (rank < static_cast<int>(n_test % num_procs) ? 1 : 0);
    std::vector<double> local_test_feats(local_n_test * NUM_FEATURES);
    std::vector<int> local_predictions(local_n_test, 0);

    // Step 3: Scatter chunks of the test queries to each rank
    MPI_Scatterv(rank == 0 ? flat_test_feats.data() : nullptr,
                 send_counts_feat.data(),
                 displs_feat.data(),
                 MPI_DOUBLE,
                 local_test_feats.data(),
                 local_n_test * NUM_FEATURES,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    // Synchronize all processes before starting execution timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Step 4: Computationally Intensive Stage - Parallel KNN Distance Computation
    for (int i = 0; i < local_n_test; ++i) {
        const double* cur_test_feat = &local_test_feats[i * NUM_FEATURES];
        local_predictions[i] = knn_predict_single(cur_test_feat,
                                                  flat_train_feats.data(),
                                                  train_labels.data(),
                                                  n_train,
                                                  k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double local_elapsed_ms = (end_time - start_time) * 1000.0;
    double max_elapsed_ms = 0.0;

    // Step 5: Find maximum execution time across all ranks (bottleneck process wall-clock time)
    MPI_Reduce(&local_elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Step 6: Gather all predicted labels back to Rank 0
    MPI_Gatherv(local_predictions.data(),
                local_n_test,
                MPI_INT,
                rank == 0 ? all_predictions.data() : nullptr,
                recv_counts_pred.data(),
                displs_pred.data(),
                MPI_INT,
                0,
                MPI_COMM_WORLD);

    // Rank 0: Evaluate metrics and display HPC performance
    if (rank == 0) {
        ClassificationMetrics metrics = evaluate_predictions(actual_test_labels, all_predictions);
        metrics.print("Parallel OpenMPI KNN Evaluation Results");

        std::cout << "\n-------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  MPI Process Count (p)        : " << num_procs << "\n";
        std::cout << "  Parallel Execution Time (Tp) : " << max_elapsed_ms << " ms (" << (max_elapsed_ms / 1000.0) << " s)\n";
        std::cout << "  Total Distance Operations    : " << (n_test * n_train) << " pairwise checks\n";
        std::cout << "-------------------------------------------------------\n";
    }

    MPI_Finalize();
    return 0;
}
