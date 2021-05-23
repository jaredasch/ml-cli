#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>
#include "optimizers/gradient_descent.h"

typedef Eigen::MatrixXd mat;

bool GradientDescent::run(mat &data, mat &labels) {
    int iters = 0;
    std::unordered_map<std::string, mat> gradient;

    double last_loss = 0;

    while (iters < max_iters) {
        gradient = obj->gradient(data, labels);
        for (std::pair<std::string, mat> p : gradient) {
            std::string param = p.first;
            mat grad = p.second;
            mat updated_param = (obj->params[param].array() - learning_rate * grad.array()).matrix();
            obj->params[param] = updated_param;
        }
        mat pred = obj->class_conditional_prob(data);
        double loss = obj->loss(pred, labels);

        if (iters != 0 && last_loss -loss < thresh) {
            std::cout << "Converged after " << iters << " iterations" << std::endl;
            return true;
        }

        last_loss = loss;
        iters++;
    }

    return false;
}