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


/**
*   Loads labels into vector of strings, expects newline separated labels in column format
*   @param path file path to load from
*   @param header whether or not to ignore the first row
*
*   @return ordered vector containing labels
*/
std::vector<std::string> loaders::load_labels(std::string path, bool is_header) {
    std::ifstream input_f;
    input_f.open(path);
    std::vector<std::string> labels;

    if (input_f.good()) {
        std::string buf;
        if (is_header) {
            getline(input_f, buf);
        }
        while (getline(input_f, buf)) {
            labels.push_back(buf);
        }
    }

    return labels;
}


/**
*   Adds bias column to matrix
*   @param matrix the matrix to add the column to (in place)
*/   
void loaders::add_bias(Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd new_col;
    new_col.setOnes(matrix.rows(), 1);
    matrix.conservativeResize(matrix.rows(), matrix.cols()+1);
    matrix.block(0, matrix.cols()-1, matrix.rows(), 1) = new_col;
}


void loaders::export_prediction(std::vector<std::string> prediction, std::string path) {
    std::ofstream out_file;
    out_file.open(path);
    for (int i = 0; i < (int) prediction.size(); i++) {
        out_file << prediction[i] << std::endl;
    }    
}

