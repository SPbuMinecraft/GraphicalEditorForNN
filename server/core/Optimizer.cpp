#include <vector>

#include "Operation.h"
#include "RandomInit.h"
#include "Tensor.h"
#include "Optimizer.h"

void OptimizerBase::append(std::vector<TensorRef>& newParams) {
    params.reserve(params.size() + newParams.size() * sizeof(TensorRef));
    params.insert(params.end(),newParams.begin(),newParams.end());
}

void OptimizerBase::step() {
    for (int i = 0; i < params.size(); i++) {
        *params[i].get().output -= lr * *params[i].get().gradient;
    }
}