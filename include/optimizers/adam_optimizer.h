#pragma once

#include <unordered_map>

#include "types.h"
#include "optimizers/first_order_optimizer.h"

class AdamOptimizer : public FirstOrderOptimizer {
    private:
        double learning_rate;
        int batch_size;
        int iterations;
        double beta_1;
        double beta_2;
        double epsilon;

        std::unordered_map<std::string, mat> first_moment;
        std::unordered_map<std::string, mat> second_moment;

    public:
        AdamOptimizer(double learning_rate, int batch_size=1, int iterations=12000, double beta_1=0.9, double beta_2=0.999, double epsilon=1e-8) :
            learning_rate{learning_rate},
            batch_size{batch_size},
            iterations{iterations},
            beta_1{beta_1},
            beta_2{beta_2},
            epsilon{epsilon}
            {}

        bool run(mat &data, mat &labels) override;
};