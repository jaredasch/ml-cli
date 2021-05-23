#pragma once

#include <iostream>

#include "classifiers/logistic_regression.h"

class GradientDescent {
    private:
        LogisticRegression* const obj;
        int max_iters;
        double thresh;
        double learning_rate;

    public: 
        GradientDescent(LogisticRegression* const obj, int max_iters, double thresh, double learning_rate) : 
            obj{obj},
            max_iters{max_iters},
            thresh{thresh},
            learning_rate{learning_rate}
            {}
            
        ~GradientDescent() {};

        bool run(mat &data, mat &labels);
};