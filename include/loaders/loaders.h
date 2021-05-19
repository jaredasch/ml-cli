#ifndef LOADER_H
#define LOADER_H

#include <armadillo>

using namespace arma;
using namespace std;

namespace loaders {
    mat load_csv(string path);    
}

#endif