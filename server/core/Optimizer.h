#pragma once

#include <vector>

#include "Tensor.h"

class OptimizerBase;

class LRScheduler {
public:
    virtual void step() = 0;
    virtual ~LRScheduler() = default;
};

class GammaScheduler : public LRScheduler {
public:
    GammaScheduler(OptimizerBase* optim, size_t stepSize, float gamma);
    virtual void step() override;

private:
    OptimizerBase* optim;
    size_t stepSize;
    float gamma;
    size_t stepAccum = 0;
};

class OptimizerBase {
    friend class GammaScheduler;
    friend class ConstScheduler;
public:
    OptimizerBase(float lr);
    void append(std::vector<TensorRef>& newParams);
    void step();

private:
    float lr;
    std::vector<TensorRef> params;
};
