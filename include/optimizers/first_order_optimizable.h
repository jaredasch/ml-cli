#pragma once

#include <unordered_map>

#include "types.h"

class FirstOrderOptimizable {
    // virtual void update_param(std::string, mat new_param) = 0;
    // virtual mat get_param(std::string) = 0;

    virtual std::unordered_map<std::string, mat> gradient(mat& data, mat &labels) const = 0;
    virtual double loss(mat& data, mat& true_labels) const = 0;
};