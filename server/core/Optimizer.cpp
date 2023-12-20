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
        if (params[i].get().gradient.has_value() && params[i].get().output.has_value()) {
            params[i].get().output.value() -= lr * params[i].get().gradient.value();
            Allocator::endSession();
        }
    }
}
