#include <vector>

#include "Operation.h"
#include "RandomInit.h"
#include "Tensor.h"
#include "Optimizer.h"

OptimizerBase::OptimizerBase(float lr) : lr(lr) {
}

void OptimizerBase::append(std::vector<TensorRef>& newParams) {
    params.reserve(params.size() + newParams.size());
    params.insert(params.end(), newParams.begin(), newParams.end());
}

void OptimizerBase::step() {
    for (int i = 0; i < params.size(); i++) {
        params[i].get().output.value() -= lr * params[i].get().gradient.value();
        Allocator::endSession();
    }
}

GammaScheduler::GammaScheduler(OptimizerBase* optim, size_t stepSize, float gamma)
    : optim{optim}, stepSize{stepSize}, gamma{gamma} {
}

void GammaScheduler::step() {
    ++stepAccum;
    if (optim && stepAccum == stepSize) {
        optim->lr *= gamma;
        stepAccum = 0;
    }
}
