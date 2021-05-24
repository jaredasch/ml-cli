#include <vector>
#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include "loaders/loaders.h"

Eigen::MatrixXd loaders::load_csv(std::string path, char separator, bool is_header) {
    std::ifstream input_f;
    input_f.open(path);

    if (input_f.good()) {
        // If header in file, get rid of it
        std::string line_buf;
        if (is_header) {
            getline(input_f, line_buf);
        };
        
        // Parse file into matrix
        unsigned int current_row = 0;
        unsigned int cols = 0;

        // Initialize 2d vector
        std::vector<std::vector<double>> temp_mat;

        while (getline(input_f, line_buf)) {
            // Add empty vector for row
            temp_mat.push_back(std::vector<double>());

            std::stringstream line_stream(line_buf);
            std::string val;

            // Read until seperator
            while (getline(line_stream, val, separator)) {
                temp_mat[current_row].push_back(stod(val));
            }

            // Check that each row has the same number of cols
            if (current_row == 0) {
                cols = temp_mat[0].size();
            } else if (temp_mat[current_row].size() != cols) {
                std::cout << "Something's wrong\n";
            }

            current_row++;
        }

        // Construct matrix from vector
        Eigen::MatrixXd loaded_mat(current_row, cols);
        for (unsigned int row = 0; row < current_row; row++) {
            for (unsigned int col = 0; col < cols; col++) {
                loaded_mat(row, col) = temp_mat[row][col];
            }
        }

        input_f.close();
        return loaded_mat;
    } else {
        // Figure out error handling later
        std::cout << "Error opening file\n";
        return Eigen::MatrixXd(3, 3);
    }
}

void loaders::add_bias(Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd new_col;
    new_col.setOnes(matrix.rows(), 1);
    matrix.conservativeResize(matrix.rows(), matrix.cols()+1);
    matrix.block(0, 0, matrix.rows(), 1) = new_col;
}