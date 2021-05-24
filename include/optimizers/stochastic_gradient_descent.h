#pragma once

#include "types.h"
#include "optimizers/first_order_optimizer.h"

class StochasticGradientDescent : public FirstOrderOptimizer {
    private:
        int epochs;
        double learning_rate;

    public:
        StochasticGradientDescent(int epochs, double learning_rate) : 
            epochs{epochs},
            learning_rate{learning_rate}
            {}

        bool run(mat &data, mat &labels) override;
};