#pragma once

#include <unordered_map>

#include "optimizers/first_order_optimizable.h"

using namespace std;
using namespace arma;

class LogisticRegression: public FirstOrderOptimizable {
    unordered_map<string, mat> params;

    public:
        LogisticRegression(int dim);

        void fit(mat &data, vec &labels);
        vec predict(mat &data);

        double loss(mat &data, vec &labels);
        unordered_map<string, mat> gradient(mat &data, vec &labels);
};