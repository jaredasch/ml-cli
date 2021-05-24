#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>
#include "optimizers/batch_gradient_descent.h"

#include "types.h"

/**
*   Minimizes the loss function of this->obj using provided gradients
*/
bool BatchGradientDescent::run(mat &data, mat &labels) {
    int iters = 0;
    std::unordered_map<std::string, mat> gradient_map;

    double last_loss = 0;

    while (iters < max_iters) {
        std::cout << "\r Iteration " << iters << std::flush;
        gradient_map = gradient(data, labels);
        for (std::pair<std::string, mat> p : gradient_map) {
            std::string param_name = p.first;
            mat grad = p.second;

            mat current_param_val = get_param(param_name);

            mat updated_param_val = (current_param_val.array() - learning_rate * grad.array()).matrix();
            
            update_param(param_name, updated_param_val);
        }

        double new_loss = loss(data, labels);

        if (iters != 0 && last_loss - new_loss < conv_thresh) {
            std::cout << "Converged after " << iters << " iterations" << std::endl;
            return true;
        }

        last_loss = new_loss;
        iters++;
    }

    std::cout << "\r" << std::flush;
    return false;
}