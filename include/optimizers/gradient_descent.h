#pragma once

#include <iostream>

#include "types.h"
#include "classifiers/logistic_regression.h"
#include "optimizers/first_order_optimizer.h"

class GradientDescent : public FirstOrderOptimizer {
    private:
        int max_iters;
        double thresh;
        double learning_rate;

    public: 
        GradientDescent(int max_iters, double thresh, double learning_rate) : 
            max_iters{max_iters},
            thresh{thresh},
            learning_rate{learning_rate}
            {}
            
        ~GradientDescent() {};

        bool run(mat &data, mat &labels) override;
};