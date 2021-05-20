#include <armadillo>

#include "loaders/loaders.h"

using namespace arma;
using namespace std;

mat loaders::load_csv(string path, char separator, bool is_header) {
    ifstream input_f;
    input_f.open(path);

    if (input_f.good()) {
        // If header in file, get rid of it
        string line_buf;
        if (is_header) {
            getline(input_f, line_buf);
        };
        
        // Parse file into matrix
        unsigned int current_row = 0;
        unsigned int cols = 0;

        // Initialize 2d vector
        vector<vector<double>> temp_mat;

        while (getline(input_f, line_buf)) {
            // Add empty vector for row
            temp_mat.push_back(vector<double>());

            stringstream line_stream(line_buf);
            string val;

            // Read until seperator
            while (getline(line_stream, val, separator)) {
                temp_mat[current_row].push_back(stod(val));
            }

            // Check that each row has the same number of cols
            if (current_row == 0) {
                cols = temp_mat[0].size();
            } else if (temp_mat[current_row].size() != cols) {
                cout << "Something's wrong\n";
            }

            current_row++;
        }

        // Construct matrix from vector
        mat loaded_mat(current_row, cols, fill::zeros);
        for (unsigned int row = 0; row < current_row; row++) {
            for (unsigned int col = 0; col < cols; col++) {
                loaded_mat(row, col) = temp_mat[row][col];
            }
        }

        input_f.close();
        return loaded_mat;
    } else {
        // Figure out error handling later
        cout << "Error opening file\n";
        return mat(3, 3, fill::zeros);
    }
}