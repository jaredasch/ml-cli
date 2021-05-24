#pragma once

#include "types.h"
#include "optimizers/first_order_optimizer.h"

class MinibatchGradientDescent : public FirstOrderOptimizer {
    private:
        int epochs;
        int batch_size;
        double learning_rate;
    public:
        MinibatchGradientDescent(int epochs, int batch_size, double learning_rate) :
            epochs{epochs},
            batch_size{batch_size},
            learning_rate{learning_rate}
            {}

        bool run(mat &data, mat &labels) override;
};