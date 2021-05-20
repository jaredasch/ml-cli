#include <iostream>
#include <armadillo>

#include "loaders/loaders.h"

#include "classifiers/logistic_regression.h"
#include "optimizers/gradient_descent.h"

using namespace std;
using namespace arma;
using namespace loaders;

int main() {
    mat train = load_csv("data/X_train.txt");

    LogisticRegression reg = LogisticRegression(train.n_cols);
    GradientDescent opt = GradientDescent(reg, 1000, 0.01);

    opt.run();
}