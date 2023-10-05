#pragma once

#include <vector>
#include <algorithm>
#include "Blob.h"
#include "Operation.h"


class Tensor;
typedef std::reference_wrapper<Tensor> TensorRef;

class Tensor {
    Blob output;
    Blob gradient;
    std::vector<TensorRef> parents;
    const Operation& operation;
    bool isOutputCached = false;
    int childrenCount = 0;
    int childrenGradReady = 0;

    Tensor(const Operation& operation, std::vector<TensorRef> parents);
    Tensor();

    Blob& forward();
    void backward();
    void accumulate(Blob& grad);

    void clear();
};
