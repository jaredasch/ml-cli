#include <unordered_map>
#include <Eigen/Dense>
#include <iostream>

#include "classifiers/binary_logistic_regression.h"

#include "types.h"

/**
*   Calculates the gradient of each parameter with respect to cross-entropy loss
*   @param data the data to calculate the gradient on
*   @param label the correct labels for the data passed in
*   @return a map string->gradient vector 
*/
std::unordered_map<std::string, mat> BinaryLogisticRegression::gradient(mat &data, mat &labels) const {
    std::unordered_map<std::string, mat> gradient;

    mat err = (class_conditional_prob(data).array() - labels.array()).matrix();
    mat gradient_w = (data.transpose() * err) / data.rows();

    gradient["w"] = gradient_w;    
    return gradient;
}

/**
*   Updates parameters
*   @param param_name the parameter to update
*   @param new_param the value for the new parameter
*/
void BinaryLogisticRegression::update_param(std::string param_name, mat new_param) {
    params[param_name] = new_param;
}

/**
*   Getter for parameters
*   @param param_name the parameter name to get
*   @return the corresponding parameter value 
*/
mat BinaryLogisticRegression::get_param(std::string param_name) const {
    return params.at(param_name);
}

/**
*   Minimizes loss on the provided data using the gradient function
*   @param data the data to fit to
*   @param labels the true labels of the data
*   @param opt optimizer object to use
*/
void BinaryLogisticRegression::fit(mat &data, mat &labels, FirstOrderOptimizer& opt) {
    // Construct optimizer
    opt.bind(this);
    opt.run(data, labels);
    opt.unbind();
    return;
}

/**
*   Calculates cross-entropy loss of the given data using the current model
*   @param data the data to calculate loss on
*   @param actual the true labels of the provided data
*   @return the cross-entropy loss
*/
double BinaryLogisticRegression::loss(mat &data, mat &actual) const {
    Eigen::ArrayXd pred_array = class_conditional_prob(data).array();
    Eigen::ArrayXd actual_array = actual.array();

    double loss =   (-1 * actual.transpose() * (log(pred_array) / log(2)).matrix()).value()
                    -((1 - actual_array).matrix().transpose()  * (log(1 - pred_array) / log(2)).matrix()).value();

    return loss;
}

/**
*   Constructor for the LogisticRegression object
*   @param dim the dimension of the data to be provided
*/
BinaryLogisticRegression::BinaryLogisticRegression(int dim) {
    params["w"] = mat::Zero(dim, 1);
} 

/**
*   Calculates element-wise sigmoid on a matrix
*   @param x input matrix
*   @return sigmoid of the input
*/
mat BinaryLogisticRegression::sigmoid(mat x) {
    return (1 / ((-1 * x.array()).exp() + 1)).matrix();
}

/**
*   The predicted label of the current model
*   @param data the data to predict
*   @return vector of predicted labels
*/
mat BinaryLogisticRegression::predict(mat &data) const {
    return class_conditional_prob(data).array().round().matrix();
}

/**
*   Calculates 0-1 loss between actual and predicted
*/
double BinaryLogisticRegression::accuracy(mat &pred, mat &actual) const {
    return (pred.array() - actual.array()).abs().matrix().sum() / pred.rows();
}

/**
*   Calculates class conditional probability Pr(y=1 | x) for each row
*   @param data the data to predict on
*   @return class-conditional probability vector
*/
mat BinaryLogisticRegression::class_conditional_prob(mat &data) const {
    return sigmoid(data * params.at("w"));
}