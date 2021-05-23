#pragma once

#include <iostream>
#include <unordered_map>

typedef Eigen::MatrixXd mat;

class LogisticRegression {
    public:
        static double sigmoid(double x);
        static mat sigmoid(mat x);

        LogisticRegression(int dim);

        void fit(mat &data, mat &labels);

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
        double loss(mat &pred, mat &actual) const;

        double accuracy(mat &pred, mat &actual) const;

        /**
        *   Calculates the gradient of the loss w.r.t each parameter
        *   @return map from parameter name -> gradient vector for that parameter
        */
        std::unordered_map<std::string, mat> gradient(mat &data, mat &labels) const;

        // Map from param name -> param vector
        std::unordered_map<std::string, mat> params;
};