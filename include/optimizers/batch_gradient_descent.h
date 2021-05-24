#pragma once

#include <iostream>

#include "types.h"
#include "classifiers/logistic_regression.h"
#include "optimizers/first_order_optimizer.h"

class BatchGradientDescent : public FirstOrderOptimizer {
    private:
        int max_iters;
        double thresh;
        double learning_rate;

    public: 
        BatchGradientDescent(int max_iters, double thresh, double learning_rate) : 
            max_iters{max_iters},
            thresh{thresh},
            learning_rate{learning_rate}
            {}
            
        bool run(mat &data, mat &labels) override;
};