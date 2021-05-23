#pragma once

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
    *   Adds bias column to matrix
    *   @param matrix the matrix to add the column to (in place)
    */   
    void add_bias(Eigen::MatrixXd& matrix);
}
