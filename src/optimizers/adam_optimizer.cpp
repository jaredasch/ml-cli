#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "types.h"
#include "optimizers/adam_optimizer.h"

bool AdamOptimizer::run(mat &data, mat &labels) {
    srand(time(0));

    // Store how many have already been sampled to reshuffle when needed
    int sampled_from_current_shuffling = 0;
    std::vector<int> shuffled_indices;
    for(int i = 0; i < data.rows(); i++) {
        shuffled_indices.push_back(i);
    }
    std::random_shuffle(shuffled_indices.begin(), shuffled_indices.end());

    // Main loop
    int t = 0;
    while (t < iterations) {
        t++;
        std::cout << "\rt = " << t << std::flush;

        // Construct minibatch, reshuffling if needed
        if (sampled_from_current_shuffling == data.rows()) {
            std::random_shuffle(shuffled_indices.begin(), shuffled_indices.end());
            sampled_from_current_shuffling = 0;
        }

        int remaining = data.rows() - sampled_from_current_shuffling;
        int current_batch_size = std::min(batch_size, remaining);

        // Put minibtach data and labels into new matrices
        mat batch_data(current_batch_size, data.cols());
        mat batch_labels(current_batch_size, 1);

        for (int i = 0; i < current_batch_size; i++) {
            batch_data.block(i, 0, 1, data.cols()) = data.block(shuffled_indices[sampled_from_current_shuffling], 0, 1, data.cols());
            batch_labels.block(i, 0, 1, 1) = labels.block(shuffled_indices[sampled_from_current_shuffling], 0, 1, 1);
            sampled_from_current_shuffling++;
        }

        std::unordered_map<std::string, mat> grad_map = gradient(batch_data, batch_labels);
        
        // Set initialization terms
        if (t == 1) {
            for (std::pair<std::string, mat> p : grad_map) {
                std::string param_name = p.first;
                mat param_grad = p.second;

                first_moment[param_name] = mat::Constant(param_grad.rows(), param_grad.cols(), 0);   
                second_moment[param_name] = mat::Constant(param_grad.rows(), param_grad.cols(), 0);              
            }
        }

        // Compute moment terms and update the gradient
        for (std::pair<std::string, mat> p : grad_map) {
            std::string param_name = p.first;
            mat grad = p.second;

            // Compute moment terms without bias correction
            mat updated_first_moment = (beta_1 * first_moment[param_name].array() + (1 - beta_1) * grad.array()).matrix();
            mat updated_second_moment = (beta_2 * second_moment[param_name].array() + (1 - beta_2) * grad.array().square()).matrix();

            first_moment[param_name] = updated_first_moment;
            second_moment[param_name] = updated_second_moment;

            // Bias correction
            mat bias_corrected_first_moment = updated_first_moment / (1 - std::pow(beta_1, t));
            mat bias_corrected_second_moment = updated_second_moment / (1 - std::pow(beta_2, t));

            // Update gradient
            mat updated_param = (get_param(param_name).array() - learning_rate * (bias_corrected_first_moment.array() / (bias_corrected_second_moment.array().sqrt() + epsilon))).matrix();

            update_param(param_name, updated_param);
        }  
    }
    std::cout << "\r" << std::flush;
    return false;
}