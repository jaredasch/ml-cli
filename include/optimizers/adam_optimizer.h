#pragma once

#include <unordered_map>

#include "types.h"
#include "optimizers/first_order_optimizer.h"

class AdamOptimizer : public FirstOrderOptimizer {
    private:
        double learning_rate;
        double conv_thresh;
        int batch_size;
        int max_iters;
        double beta_1;
        double beta_2;
        double epsilon;

        std::unordered_map<std::string, mat> first_moment;
        std::unordered_map<std::string, mat> second_moment;

    public:
        AdamOptimizer(double learning_rate, double conv_thresh, int batch_size=1, int max_iters=12000, double beta_1=0.9, double beta_2=0.999, double epsilon=1e-8) :
            learning_rate{learning_rate},
            conv_thresh{conv_thresh},
            batch_size{batch_size},
            max_iters{max_iters},
            beta_1{beta_1},
            beta_2{beta_2},
            epsilon{epsilon}
            {}

        bool run(mat &data, mat &labels) override;
};