#pragma once

#include <optional>

#include "Operation.h"
#include <optional>

class Tensor;
typedef std::reference_wrapper<Tensor> TensorRef;


class Tensor {
private:
    bool outputIsNull = true;
    int childrenCount = 0;
    int childrenGradReady = 0;
    const Operation& operation;
    std::vector<TensorRef> parents;

    void getParentsData(std::vector<BlobRef> &datas);
    void getParentsGrads(std::vector<BlobRef> &grads);

public:
    std::optional<Blob> output;
    std::optional<Blob> gradient;

    Tensor(const Operation& operation, const std::vector<TensorRef>& parents, bool noGrad = false);
    Tensor(const Blob& data);

    Tensor operator=(const Tensor & other);

    BlobRef forward();
    void backward();
    void accumulate();

    void clear();
};
