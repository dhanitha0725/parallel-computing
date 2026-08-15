#include "common.hpp"

int main(int argc, char* argv[]) {
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

    std::cout << "=======================================================\n";
    std::cout << "       Sequential K-Nearest Neighbors (KNN) in C++     \n";
    std::cout << "=======================================================\n";
    std::cout << "Dataset Path : " << dataset_path << "\n";
    std::cout << "Sample Limit : " << sample_size << " records (0.5% SUSY subset)\n";
    std::cout << "K Value      : " << k << "\n";

    // 1. Load Dataset
    std::cout << "\n[1/4] Loading dataset from disk..." << std::endl;
    std::vector<Event> full_dataset;
    if (!load_susy_csv(dataset_path, full_dataset, sample_size)) {
        std::cerr << "Failed to load dataset from " << dataset_path << std::endl;
        return 1;
    }
    std::cout << "Successfully loaded " << full_dataset.size() << " collision events." << std::endl;

    // 2. Train/Test Split (80% Train, 20% Test)
    std::cout << "[2/4] Partitioning into 80% Train / 20% Test..." << std::endl;
    std::vector<Event> train_events, test_events;
    train_test_split(full_dataset, train_events, test_events, 0.80);
    std::cout << "Training samples : " << train_events.size() << "\n";
    std::cout << "Testing samples  : " << test_events.size() << std::endl;

    // 3. Feature Standardization
    std::cout << "[3/4] Standardizing feature values (Z-Score)..." << std::endl;
    standardize_dataset(train_events, test_events);

    // Prepare contiguous arrays for maximum memory locality & cache efficiency
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

    std::vector<int> predictions(n_test, 0);

    // 4. Computationally Intensive Stage: Sequential KNN Classification
    std::cout << "\n[4/4] Executing Sequential KNN Classification (Computing distances)..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n_test; ++i) {
        const double* cur_test_feat = &flat_test_feats[i * NUM_FEATURES];
        predictions[i] = knn_predict_single(cur_test_feat,
                                            flat_train_feats.data(),
                                            train_labels.data(),
                                            n_train,
                                            k);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double exec_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double exec_time_sec = exec_time_ms / 1000.0;

    // 5. Evaluate Results
    ClassificationMetrics metrics = evaluate_predictions(actual_test_labels, predictions);
    metrics.print("Sequential KNN Evaluation Results");

    std::cout << "\n-------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Sequential Execution Time (Ts) : " << exec_time_ms << " ms (" << exec_time_sec << " s)\n";
    std::cout << "  Processed Distance Operations   : " << (n_test * n_train) << " pairwise checks\n";
    std::cout << "-------------------------------------------------------\n";

    return 0;
}
