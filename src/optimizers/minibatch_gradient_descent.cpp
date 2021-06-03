#include <iostream>
#include <vector>
#include <algorithm>

#include <Eigen/Dense>
#include "optimizers/minibatch_gradient_descent.h"

#include "types.h"

/**
*   Minimizes the loss function of this->obj using provided gradients
*/
bool MinibatchGradientDescent::run(mat &data, mat &labels) {
    srand(time(0));
    std::unordered_map<std::string, mat> gradient_map;

    for (int epoch = 0; epoch < epochs; epoch++) {

        // Construct random permutation for minibatch sampling
        std::vector<int> shuffled_indices;
        for(int i = 0; i < data.rows(); i++) {
            shuffled_indices.push_back(i);
        }
        std::random_shuffle(shuffled_indices.begin(), shuffled_indices.end());

        // While still more left to sample
        int sampled = 0;
        while (sampled < data.rows()) {
            int remaining = data.rows() - sampled;
            int current_batch_size = std::min(batch_size, remaining);

            mat batch_data(current_batch_size, data.cols());
            mat batch_labels(current_batch_size, 1);

            for (int i = 0; i < current_batch_size; i++) {
                batch_data.block(i, 0, 1, data.cols()) = data.block(shuffled_indices[sampled], 0, 1, data.cols());
                batch_labels.block(i, 0, 1, 1) = labels.block(shuffled_indices[sampled], 0, 1, 1);
                sampled++;
            }

            gradient_map = gradient(batch_data, batch_labels);
            for (std::pair<std::string, mat> p : gradient_map) {
                std::string param_name = p.first;
                mat grad = p.second;
                mat current_param_val = get_param(param_name);
                mat updated_param_val = (current_param_val.array() - learning_rate * grad.array()).matrix();   
                update_param(param_name, updated_param_val);
            }
        }
    }

    return true;
}