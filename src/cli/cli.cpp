#include <iostream>
#include <Eigen/Dense>

#include "types.h"

#include "loaders/loaders.h"
#include "classifiers/logistic_regression.h"
#include "optimizers/gradient_descent.h"

int main() {
    mat x_train = loaders::load_csv("data/X_train.txt");
    mat y_train = loaders::load_csv("data/y_train.txt");

    loaders::add_bias(x_train);

    LogisticRegression reg = LogisticRegression(x_train.cols());
    GradientDescent opt = GradientDescent(10000, 0.0001, 1);

    reg.fit(x_train, y_train, opt);

    mat pred_train = reg.predict(x_train);
    std::cout << "Train accuracy: " << reg.accuracy(pred_train, y_train) << std::endl;

    mat x_test = loaders::load_csv("data/X_test.txt");
    mat y_test = loaders::load_csv("data/y_test.txt");

    loaders::add_bias(x_test);
    mat pred_test = reg.predict(x_test);
    std::cout << "Test accuracy: " << reg.accuracy(pred_test, y_test) << std::endl;
}