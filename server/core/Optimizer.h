#pragma once

#include <vector>

#include "Tensor.h"

class OptimizerBase {
public:
    OptimizerBase(float lr);
    void append(std::vector<TensorRef>& newParams);
    void step();

private:
    const float lr;
    std::vector<TensorRef> params;
};
