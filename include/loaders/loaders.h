#ifndef LOADER_H
#define LOADER_H

#include <armadillo>

using namespace arma;
using namespace std;

/**
*   Loads CSV from a file into a mat type 
*   @param path the file path to load from
*   @param separator the separator used in the file, defaults to a comma
*   @param header whether or not to ignore the first row
*
*   @return the matrix loaded from the file
*/

namespace loaders {
    mat load_csv(string path, char separator=',', bool is_header=false);    
}

#endif