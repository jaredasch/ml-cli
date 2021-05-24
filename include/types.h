#pragma once

#include <Eigen/Dense>

typedef Eigen::MatrixXd mat;

// First order optimizable
class FirstOrderOptimizable;
class LogisticRegression;

// First order optimizers
class FirstOrderOptimizer;
class BatchGradientDescent;
class StochasticGradientDescent;
