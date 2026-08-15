#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <queue>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <numeric>

constexpr int NUM_FEATURES = 18;
constexpr int DEFAULT_K = 5;

// Struct to store a single particle-collision event
struct Event {
    int label; // 0 = Background, 1 = SUSY Signal
    std::vector<double> features;

    Event() : label(0), features(NUM_FEATURES, 0.0) {}
    Event(int lbl, const std::vector<double>& feat) : label(lbl), features(feat) {}
};

// Struct to store confusion matrix and classification metrics
struct ClassificationMetrics {
    int total_samples = 0;
    int true_positives = 0;   // Predicted 1, Actual 1
    int true_negatives = 0;   // Predicted 0, Actual 0
    int false_positives = 0;  // Predicted 1, Actual 0
    int false_negatives = 0;  // Predicted 0, Actual 1

    double accuracy() const {
        return total_samples > 0 ? (double)(true_positives + true_negatives) / total_samples : 0.0;
    }

    double precision() const {
        int denom = true_positives + false_positives;
        return denom > 0 ? (double)true_positives / denom : 0.0;
    }

    double recall() const {
        int denom = true_positives + false_negatives;
        return denom > 0 ? (double)true_positives / denom : 0.0;
    }

    double f1_score() const {
        double p = precision();
        double r = recall();
        return (p + r > 0.0) ? 2.0 * (p * r) / (p + r) : 0.0;
    }

    void print(const std::string& title = "Evaluation Metrics") const {
        std::cout << "\n=======================================================\n";
        std::cout << "           " << title << "\n";
        std::cout << "=======================================================\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Total Test Samples   : " << total_samples << "\n";
        std::cout << "  True Positives (TP)  : " << true_positives << "\n";
        std::cout << "  True Negatives (TN)  : " << true_negatives << "\n";
        std::cout << "  False Positives (FP) : " << false_positives << "\n";
        std::cout << "  False Negatives (FN) : " << false_negatives << "\n";
        std::cout << "  ---------------------------------------------------\n";
        std::cout << "  Accuracy             : " << (accuracy() * 100.0) << " %\n";
        std::cout << "  Precision            : " << precision() << "\n";
        std::cout << "  Recall               : " << recall() << "\n";
        std::cout << "  F1-Score             : " << f1_score() << "\n";
        std::cout << "=======================================================\n";
    }
};

// Calculate squared Euclidean distance between two feature vectors
inline double squared_euclidean_distance(const double* a, const double* b, int dim = NUM_FEATURES) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Function to load the SUSY dataset (with optional max_rows limit)
inline bool load_susy_csv(const std::string& filename,
                          std::vector<Event>& dataset,
                          size_t max_rows = 25000) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    dataset.clear();
    dataset.reserve(max_rows);

    std::string line;
    bool is_first_line = true;
    size_t count = 0;

    while (std::getline(file, line) && count < max_rows) {
        if (line.empty()) continue;

        // Skip header if first line contains column names
        if (is_first_line) {
            is_first_line = false;
            if (line.find("SUSY") != std::string::npos || line.find("lepton") != std::string::npos) {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string token;
        
        // Read label (first column)
        if (!std::getline(ss, token, ',')) continue;
        double raw_label = std::stod(token);
        int label = (raw_label >= 0.5) ? 1 : 0;

        // Read 18 numerical features
        std::vector<double> features(NUM_FEATURES);
        bool valid_row = true;
        for (int i = 0; i < NUM_FEATURES; ++i) {
            if (std::getline(ss, token, ',')) {
                try {
                    features[i] = std::stod(token);
                } catch (...) {
                    valid_row = false;
                    break;
                }
            } else {
                valid_row = false;
                break;
            }
        }

        if (valid_row) {
            dataset.emplace_back(label, features);
            count++;
        }
    }

    file.close();
    return !dataset.empty();
}

// Standardization: (x - mean) / std_dev for each feature
inline void standardize_dataset(std::vector<Event>& train_data, std::vector<Event>& test_data) {
    if (train_data.empty()) return;

    std::vector<double> means(NUM_FEATURES, 0.0);
    std::vector<double> std_devs(NUM_FEATURES, 0.0);
    size_t n_train = train_data.size();

    // 1. Calculate Mean
    for (const auto& ev : train_data) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            means[j] += ev.features[j];
        }
    }
    for (int j = 0; j < NUM_FEATURES; ++j) {
        means[j] /= n_train;
    }

    // 2. Calculate Standard Deviation
    for (const auto& ev : train_data) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            double diff = ev.features[j] - means[j];
            std_devs[j] += diff * diff;
        }
    }
    for (int j = 0; j < NUM_FEATURES; ++j) {
        std_devs[j] = std::sqrt(std_devs[j] / n_train);
        if (std_devs[j] < 1e-9) std_devs[j] = 1.0; // Avoid division by zero
    }

    // 3. Apply to Train Set
    for (auto& ev : train_data) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            ev.features[j] = (ev.features[j] - means[j]) / std_devs[j];
        }
    }

    // 4. Apply same scaling to Test Set
    for (auto& ev : test_data) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            ev.features[j] = (ev.features[j] - means[j]) / std_devs[j];
        }
    }
}

// Split dataset into train and test subsets (e.g. 80% train, 20% test)
inline void train_test_split(const std::vector<Event>& full_dataset,
                             std::vector<Event>& train_data,
                             std::vector<Event>& test_data,
                             double train_ratio = 0.8) {
    size_t total = full_dataset.size();
    size_t train_size = static_cast<size_t>(total * train_ratio);

    train_data.assign(full_dataset.begin(), full_dataset.begin() + train_size);
    test_data.assign(full_dataset.begin() + train_size, full_dataset.end());
}

// Predict single test sample using KNN
inline int knn_predict_single(const double* test_feat,
                              const double* train_feats,
                              const int* train_labels,
                              size_t n_train,
                              int k) {
    // Max-heap storing pairs of (squared_distance, label)
    // The top of the max-heap has the largest distance among current top-k
    std::priority_queue<std::pair<double, int>> max_heap;

    for (size_t i = 0; i < n_train; ++i) {
        const double* cur_train_feat = train_feats + (i * NUM_FEATURES);
        double dist = squared_euclidean_distance(test_feat, cur_train_feat, NUM_FEATURES);

        if (static_cast<int>(max_heap.size()) < k) {
            max_heap.push({dist, train_labels[i]});
        } else if (dist < max_heap.top().first) {
            max_heap.pop();
            max_heap.push({dist, train_labels[i]});
        }
    }

    // Majority voting
    int count_pos = 0;
    int count_neg = 0;
    while (!max_heap.empty()) {
        if (max_heap.top().second == 1) count_pos++;
        else count_neg++;
        max_heap.pop();
    }

    return (count_pos >= count_neg) ? 1 : 0;
}

// Evaluate metrics between true labels and predictions
inline ClassificationMetrics evaluate_predictions(const std::vector<int>& actual,
                                                  const std::vector<int>& predicted) {
    ClassificationMetrics m;
    m.total_samples = static_cast<int>(actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (predicted[i] == 1 && actual[i] == 1) m.true_positives++;
        else if (predicted[i] == 0 && actual[i] == 0) m.true_negatives++;
        else if (predicted[i] == 1 && actual[i] == 0) m.false_positives++;
        else if (predicted[i] == 0 && actual[i] == 1) m.false_negatives++;
    }
    return m;
}

#endif // COMMON_HPP
