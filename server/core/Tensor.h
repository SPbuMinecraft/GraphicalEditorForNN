#pragma once

#include "Operation.h"
#include <optional>

class Tensor;
typedef std::reference_wrapper<Tensor> TensorRef;

class Tensor {
private:
    int childrenCount = 0;
    int childrenGradReady = 0;
    const Operation& operation;
    const std::vector<TensorRef> parents;

    void getParentsData(std::vector<BlobRef> &datas);
public:
    std::optional<Blob> output;
    std::optional<Blob> gradient;

    Tensor(const Operation& operation, const std::vector<TensorRef>& parents);
    Tensor(const Blob& data);

    BlobRef forward();
    void backward();
    void accumulate(const Blob& grad);

    void clear();
};
