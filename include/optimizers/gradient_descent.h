#pragma once

#include "optimizers/first_order_optimizable.h"

class GradientDescent {
    private:
        FirstOrderOptimizable &obj;

        int max_iters;
        double thresh;
        double learning_rate;

    public: 
        GradientDescent(FirstOrderOptimizable &obj, int max_iters, double thresh, double learning_rate) : 
            obj{obj},
            max_iters{max_iters},
            thresh{thresh},
            learning_rate{learning_rate}
            {}
            
        ~GradientDescent() {}

        /**
        * Runs gradient descent on the object passed in
        * @return boolean on whether or not results converged
        */ 
        bool run();    
};