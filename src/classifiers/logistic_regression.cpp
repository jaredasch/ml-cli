#include <armadillo>
#include <unordered_map>

#include "classifiers/logistic_regression.h"

using namespace std;
using namespace arma;

unordered_map<string, mat> LogisticRegression::gradient(mat &data, vec &labels) {
    unordered_map<string, mat> ret;
    return ret;
}

double LogisticRegression::loss(mat &data, vec &labels) {
    return 0;
}

LogisticRegression::LogisticRegression(int dim) {
    params["w"] = zeros<mat>(dim+1, 0);
}