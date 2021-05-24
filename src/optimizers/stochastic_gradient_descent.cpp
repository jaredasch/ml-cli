#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include <Eigen/Dense>
#include "optimizers/stochastic_gradient_descent.h"

#include "types.h"

/**
*   Minimizes the loss function of this->obj using provided gradients
*/
bool StochasticGradientDescent::run(mat &data, mat &labels) {
    srand(time(0));
    std::unordered_map<std::string, mat> gradient_map;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "\r Epoch " << epoch << std::flush;

        // Construct random permutation for minibatch sampling
        std::vector<int> shuffled_indices;
        for(int i = 0; i < data.rows(); i++) {
            shuffled_indices.push_back(i);
        }
        std::random_shuffle(shuffled_indices.begin(), shuffled_indices.end());

        for(int example = 0; example < data.rows(); example++) {
            mat data_row = data.block(shuffled_indices[example], 0, 1, data.cols());
            mat label_row = labels.block(shuffled_indices[example], 0, 1, 1);
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