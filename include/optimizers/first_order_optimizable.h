#pragma once

#include <armadillo>
#include <unordered_map>

using namespace std;
using namespace arma;

class FirstOrderOptimizable {
    public: 
        unordered_map<string, mat> params;

        FirstOrderOptimizable() {}
        virtual ~FirstOrderOptimizable() {}

        /**
        *   Computes the gradient of the loss for each parameter in the map on the given data
        */
        virtual unordered_map<string, mat> gradient(mat &data, vec &labels) = 0;

        /**
        *   Computes the loss on the given data
        */
        virtual double loss(mat &data, vec &labels) = 0;
};