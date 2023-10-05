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

//кажется, что нужно добавить Blob& data в конструктор, иначе какова размерность и значение output и gradient?
Tensor::Tensor(const Operation& operation, vector<TensorRef> parents): operation(operation) {
    this->parents = parents;
    std::transform(parents.begin(), parents.end(), parents.begin(),
                    [](TensorRef t) { t.get().childrenCount++; });
    output = Blob((size_t)1, (size_t)1);
    gradient = Blob((size_t)1, (size_t)1);
}

Tensor::Tensor(): Tensor(noop, vector<TensorRef> {}) {};

Blob& Tensor::forward() {
    vector<BlobRef> datas((size_t)parents.size());
    std::transform(parents.begin(), parents.end(), datas.begin(), 
                    [](TensorRef t) { return t.get().output; });
    output = operation.compute(datas);
    return output;
}
void Tensor::backward(){
    vector<BlobRef> datas((size_t)parents.size());
    std::transform(parents.begin(), parents.end(), datas.begin(),
                    [](TensorRef t) { return t.get().output; });
    vector<BlobRef> gs = operation.grad(gradient, datas);
    assert(gs.size() == parents.size());
    for (int i = 0; i < parents.size(); ++i) parents[i].get().accumulate(gs[i]);

}

void Tensor::accumulate(Blob&  grad){
    gradient += grad;
    std::transform(parents.begin(), parents.end(), parents.begin(),
                    [](TensorRef t) { t.get().childrenGradReady++; });
    for(auto p: parents) 
        if (p.get().childrenGradReady == p.get().childrenCount)
            p.get().accumulate(grad);
};

void Tensor::clear(){
    output = Blob((size_t)1, (size_t)1);
    gradient = Blob((size_t)1, (size_t)1);
    this->isOutputCached = false;
    for (auto p: parents) p.get().clear();
};