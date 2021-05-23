#include <unordered_map>
#include <Eigen/Dense>
#include <iostream>

#include "classifiers/logistic_regression.h"
#include "optimizers/gradient_descent.h"

typedef Eigen::MatrixXd mat;
typedef Eigen::VectorXd vec;

std::unordered_map<std::string, mat> LogisticRegression::gradient(mat &data, mat &labels) const {
    std::unordered_map<std::string, mat> gradient;

    mat err = (class_conditional_prob(data).array() - labels.array()).matrix();
    mat gradient_w = (data.transpose() * err) / data.rows();

    gradient["w"] = gradient_w;    
    return gradient;
}

double LogisticRegression::sigmoid(double x) {
    return 1.0 / (1 + std::exp(-x));
}

mat LogisticRegression::sigmoid(mat x) {
    return (1 / ((-1 * x.array()).exp() + 1)).matrix();
}

mat LogisticRegression::predict(mat &data) const {
    return class_conditional_prob(data).array().round().matrix();
}

double LogisticRegression::accuracy(mat &pred, mat &actual) const {
    return (pred.array() - actual.array()).abs().matrix().sum() / pred.rows();
}

mat LogisticRegression::class_conditional_prob(mat &data) const {
    return sigmoid(data * params.at("w"));
}

void LogisticRegression::fit(mat &data, mat &labels) {
    // Construct optimizer
    GradientDescent* opt = new GradientDescent(this, 100000, 0, 5);
    opt->run(data, labels);
    delete opt;
    return;
}

double LogisticRegression::loss(mat &pred, mat &actual) const {
    Eigen::ArrayXd pred_array = pred.array();
    Eigen::ArrayXd actual_array = actual.array();

    double loss =   (-1 * actual.transpose() * (log(pred_array) / log(2)).matrix()).value()
                    -((1 - actual_array).matrix().transpose()  * (log(1 - pred_array) / log(2)).matrix()).value();

    return loss;
}

LogisticRegression::LogisticRegression(int dim) {
    params["w"] = mat::Zero(dim, 1);
} 