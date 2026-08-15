#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char* argv[]) {
    std::string input_file = "data/supersymmetry_dataset.csv";
    std::string output_file = "data/susy_0.5percent.csv";
    size_t target_rows = 25000;

    if (argc >= 2) input_file = argv[1];
    if (argc >= 3) output_file = argv[2];
    if (argc >= 4) target_rows = std::stoull(argv[3]);

    std::cout << "Extracting " << target_rows << " records (0.5% subset) from " << input_file << " -> " << output_file << "..." << std::endl;

    std::ifstream in(input_file);
    if (!in.is_open()) {
        std::cerr << "Error: Cannot open " << input_file << std::endl;
        return 1;
    }

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot create " << output_file << std::endl;
        return 1;
    }

    std::string line;
    size_t count = 0;
    bool is_header = true;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        if (is_header) {
            out << line << "\n";
            is_header = false;
            // Check if first line was a header
            if (line.find("SUSY") != std::string::npos || line.find("lepton") != std::string::npos) {
                continue;
            }
        }

        out << line << "\n";
        count++;
        if (count >= target_rows) break;
    }

    in.close();
    out.close();

    std::cout << "Successfully generated " << output_file << " with " << count << " data records." << std::endl;
    return 0;
}
