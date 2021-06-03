#pragma once

#include "types.h"
#include "optimizers/first_order_optimizable.h"

class NeuralNetwork : public FirstOrderOptimizable {
    public:
        void update_param(std::string param_name, mat new_param) override;
        mat get_param(std::string param_name) const override;

        static mat sigmoid(mat x);

        // Map from param name -> param vector
        std::unordered_map<std::string, mat> params;
}