#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>
#include "optimizers/gradient_descent.h"

#include "types.h"

/**
*   Minimizes the loss function of this->obj using provided gradients
*/
bool GradientDescent::run(mat &data, mat &labels) {
    int iters = 0;
    std::unordered_map<std::string, mat> gradient;

    double last_loss = 0;

    while (iters < max_iters) {
        gradient = obj->gradient(data, labels);
        for (std::pair<std::string, mat> p : gradient) {
            std::string param_name = p.first;
            mat grad = p.second;
            mat current_param_val = obj->get_param(param_name);

            mat updated_param_val = (current_param_val.array() - learning_rate * grad.array()).matrix();
            obj->update_param(param_name, updated_param_val);
        }

        double loss = obj->loss(data, labels);

        if (iters != 0 && last_loss -loss < thresh) {
            std::cout << "Converged after " << iters << " iterations" << std::endl;
            return true;
        }

        last_loss = loss;
        iters++;
    }

    return false;
}