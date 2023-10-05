#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <cstddef>
#include "Blob.h"
#include "Operation.h"
#include "Tensor.h"

using namespace std;

static None noop = None();

void get_parent_datas(vector<BlobRef> &datas, Tensor& tensor){
    std::transform(tensor.parents.begin(), tensor.parents.end(), datas.begin(),
                [](TensorRef t) { return t.get().output; });
}

Tensor::Tensor(const Operation& operation, vector<TensorRef> parents): operation(operation) {
    this->parents = parents;
    std::transform(parents.begin(), parents.end(), parents.begin(),
                    [](TensorRef t) { t.get().childrenCount++; });

    vector<BlobRef> datas((size_t)parents.size());
    get_parent_datas(datas, *this);

    output = operation.compute(datas);
    gradient = Blob((size_t)output.rows, (size_t)output.cols);
}

Tensor::Tensor(Blob& data): operation(noop) {
    parents = vector<TensorRef> {};
    output = data;
    gradient = Blob((size_t)output.rows, (size_t)output.cols);
};

Blob& Tensor::forward() {
    vector<BlobRef> datas((size_t)parents.size());
    get_parent_datas(datas, *this);
    output = operation.compute(datas);
    return output;
}

void Tensor::backward(){
    vector<BlobRef> datas((size_t)parents.size());
    get_parent_datas(datas, *this);
    vector<BlobRef> gs = operation.grad(gradient, datas);
    assert(gs.size() == parents.size());
    for (int i = 0; i < parents.size(); ++i) parents[i].get().accumulate(gs[i]);

}

void Tensor::accumulate(Blob&  gradient){
    this->gradient += gradient;
    childrenGradReady++;
    if (childrenGradReady == childrenCount)
        backward();
};

void Tensor::clear(){
    gradient = Blob((size_t)output.rows, (size_t)output.cols);
    this->isOutputCached = false;
    for (auto p: parents) p.get().clear();
};
