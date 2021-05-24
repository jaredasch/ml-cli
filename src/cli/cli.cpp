#include <iostream>
#include <Eigen/Dense>

#include "types.h"

#include "loaders/loaders.h"
#include "classifiers/binary_logistic_regression.h"

#include "optimizers/batch_gradient_descent.h"
#include "optimizers/stochastic_gradient_descent.h"
#include "optimizers/minibatch_gradient_descent.h"
#include "optimizers/adam_optimizer.h"

int main() {
    mat x_train = loaders::load_csv("data/ex1/X_test.txt");
    mat y_train = loaders::load_csv("data/ex1/y_test.txt");

    loaders::add_bias(x_train);

    BinaryLogisticRegression reg = BinaryLogisticRegression(x_train.cols());

    // BatchGradientDescent opt = BatchGradientDescent(5000, 0, 0.1);
    // StochasticGradientDescent opt = StochasticGradientDescent(50, 0.1);
    // MinibatchGradientDescent opt = MinibatchGradientDescent(20, 1, 0.1);
    AdamOptimizer opt = AdamOptimizer(0.1, 100, 1000);

    reg.fit(x_train, y_train, opt);

    mat pred_train = reg.predict(x_train);
    std::cout << "Train accuracy: " << reg.accuracy(pred_train, y_train) << std::endl;

    mat x_test = loaders::load_csv("data/ex1/X_train.txt");
    mat y_test = loaders::load_csv("data/ex1/y_train.txt");

    loaders::add_bias(x_test);
    mat pred_test = reg.predict(x_test);
    std::cout << "Test accuracy: " << reg.accuracy(pred_test, y_test) << std::endl;
}