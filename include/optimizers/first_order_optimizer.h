#pragma once 

#include "types.h"

#include "optimizers/first_order_optimizable.h"

class FirstOrderOptimizer {
    private:
        FirstOrderOptimizable* obj;

    public:
        FirstOrderOptimizer() : obj{NULL} {}

        virtual bool run(mat &data, mat &labels) = 0;

        void bind(FirstOrderOptimizable *obj) {
            if (this->obj == NULL) {
                this->obj = obj;
            }
        }

        void unbind() {
            this->obj = NULL;
        }

        void update_param(std::string param_name, mat new_param) {
            return obj->update_param(param_name, new_param);
        }

        mat get_param(std::string param_name) const {
            return obj->get_param(param_name);
        }

        std::unordered_map<std::string, mat> gradient(mat& data, mat &labels) const {
            return obj->gradient(data, labels);
        }

        double loss(mat& data, mat& true_labels) const {
            return obj->loss(data, true_labels);
        }
};