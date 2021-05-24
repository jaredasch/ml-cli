#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>
#include "optimizers/stochastic_gradient_descent.h"

#include "types.h"

/**
*   Minimizes the loss function of this->obj using provided gradients
*/
bool StochasticGradientDescent::run(mat &data, mat &labels) {
    std::unordered_map<std::string, mat> gradient_map;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\r Epoch " << epoch << std::flush;
        for(int example = 0; example < data.rows(); example++) {
            mat data_row = data.block(example, 0, 1, data.cols());
            mat label_row = labels.block(example, 0, 1, 1);
            std::cout << label_row;
            gradient_map = gradient(data_row, label_row);

            for (std::pair<std::string, mat> p : gradient_map) {
                std::string param_name = p.first;
                mat grad = p.second;

                mat current_param_val = get_param(param_name);

                mat updated_param_val = (current_param_val.array() - learning_rate * grad.array()).matrix();
                
                update_param(param_name, updated_param_val);
            }
        }
    }

    std::cout << std::endl;
    return true;
}