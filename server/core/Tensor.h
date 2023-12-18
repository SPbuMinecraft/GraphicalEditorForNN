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
    std::vector<TensorRef> parents;

    void getParentsData(std::vector<LazyBlobRef> &datas);
    void getParentsGrads(std::vector<LazyBlobRef> &grads);
public:
    std::optional<Blob> output;
    std::optional<Blob> gradient;

    Tensor(const Operation& operation, const std::vector<TensorRef>& parents);
    /// Carefull, Blob is moved here to Tensor's ownership
    Tensor(const Blob data);

    Tensor(const Tensor& other) = delete;
    Tensor(Tensor&& other) noexcept;

    Tensor& operator = (Tensor&& other) noexcept;
//    Tensor& operator=(const Tensor & other);

    void forward();
    void backward();
    void accumulate(const LazyBlob& gradient);

    void clear();
};
