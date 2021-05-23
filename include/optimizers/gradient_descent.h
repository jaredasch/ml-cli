#pragma once

#include <iostream>

#include "types.h"

#include "classifiers/logistic_regression.h"


class GradientDescent {
    private:
        FirstOrderOptimizable* obj;
        int max_iters;
        double thresh;
        double learning_rate;

    public: 
        GradientDescent(int max_iters, double thresh, double learning_rate) : 
            obj{NULL},
            max_iters{max_iters},
            thresh{thresh},
            learning_rate{learning_rate}
            {}
            
        ~GradientDescent() {};

        void bind(FirstOrderOptimizable *obj) {
            if (this->obj == NULL) {
                this->obj = obj;
            }
        }

        void unbind() {
            this->obj = NULL;
        }

        bool run(mat &data, mat &labels);
};