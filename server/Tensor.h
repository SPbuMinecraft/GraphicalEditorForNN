#pragma once

#include <vector>
#include <algorithm>
#include "Blob.h"
#include "Operation.h"


class Tensor;
typedef std::reference_wrapper<Tensor> TensorRef;

class Tensor {
    private:
        Blob output;
        Blob gradient;
        std::vector<TensorRef> parents;
        const Operation& operation;
        bool isOutputCached = false;
        int childrenCount = 0;
        int childrenGradReady = 0;
        friend void get_parent_datas(std::vector<BlobRef> &datas, Tensor& tensor);

    public:
        Tensor(const Operation& operation, std::vector<TensorRef> parents);
        Tensor(Blob& data);
        Blob& forward();
        void backward();
        void accumulate(Blob& grad);
        void clear();
};
