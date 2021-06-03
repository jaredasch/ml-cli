#pragma once

#include <vector>
#include <Eigen/Dense>

namespace loaders {    
    /**
    *   Loads CSV from a file into a mat type 
    *   @param path the file path to load from
    *   @param separator the separator used in the file, defaults to a comma
    *   @param header whether or not to ignore the first row
    *
    *   @return the matrix loaded from the file
    */
    Eigen::MatrixXd load_csv(std::string path, char separator=',', bool is_header=false);

    /**
    *   Loads labels into vector of strings, expects newline separated labels in column format
    *   @param path file path to load from
    *   @param header whether or not to ignore the first row
    *
    *   @return ordered vector containing labels
    */
    std::vector<std::string> load_labels(std::string path, bool is_header=false);

    /**
    *   Adds bias column to matrix
    *   @param matrix the matrix to add the column to (in place)
    */   
    void add_bias(Eigen::MatrixXd& matrix);

    /**
    *   Exports prediction vector to file
    */
    void export_prediction(std::vector<std::string> prediction, std::string path);
}
