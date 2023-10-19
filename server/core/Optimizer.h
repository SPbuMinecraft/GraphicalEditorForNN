#pragma once

#include <vector>

#include "Tensor.h"

class OptimizerBase {

const float lr;
std::vector<TensorRef> params;

public:
    OptimizerBase(float lr) : lr(lr), params() {};
    void append(std::vector<TensorRef>& newParams);
    void step();
};