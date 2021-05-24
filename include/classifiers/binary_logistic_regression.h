#pragma once

#include <iostream>
#include <unordered_map>

#include "types.h"
#include "optimizers/first_order_optimizable.h"
#include "optimizers/batch_gradient_descent.h"

class BinaryLogisticRegression : public FirstOrderOptimizable {
    public:
        void update_param(std::string param_name, mat new_param) override;
        mat get_param(std::string param_name) const override;

        static mat sigmoid(mat x);

        BinaryLogisticRegression(int dim);

        void fit(mat &data, mat &labels, FirstOrderOptimizer& opt);

        /**
        *   Calculates class conditional probabilities of the given data
        *   @return vector eta where eta[i] is the probability that the ith example is in class +1
        */
        mat class_conditional_prob(mat &data) const;

        /**
        *   Predicts the class of the given data 
        */
        mat predict(mat &data) const;

        /**
        *   Calculate the loss of current model on training data
        */
        double loss(mat &pred, mat &actual) const override;

        double accuracy(mat &pred, mat &actual) const;

        /**
        *   Calculates the gradient of the loss w.r.t each parameter
        *   @return map from parameter name -> gradient vector for that parameter
        */
        std::unordered_map<std::string, mat> gradient(mat &data, mat &labels) const override;

        // Map from param name -> param vector
        std::unordered_map<std::string, mat> params;
};