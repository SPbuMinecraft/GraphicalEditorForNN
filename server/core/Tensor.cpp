#include "Tensor.h"
#include "Allocator.h"
#include <algorithm>
#include <functional>
#include <cassert>

using namespace std;

static const Noop noop = Noop();

void Tensor::getParentsData(vector<LazyBlobRef> &datas) {
    for (auto p: parents) datas.push_back(p.get().forward());
}

void Tensor::getParentsGrads(vector<LazyBlobRef> &grads) {
    for (auto p: parents) grads.push_back(*p.get().gradient);
}

Tensor::Tensor(Operation& operation, const vector<TensorRef>& parents): operation(operation), parents(parents) {
    for (auto p: parents) p.get().childrenCount++;
}

Tensor::Tensor(Blob data): operation(noop), parents({}) {
    this->output = std::move(data);
}

Tensor::Tensor(Tensor&& other) noexcept: operation(other.operation) {
    swap(this->parents, other.parents);
    swap(this->output, other.output);
    swap(this->gradient, other.gradient);
    swap(this->childrenCount, other.childrenCount);
    swap(this->childrenGradReady, other.childrenGradReady);
}

Tensor& Tensor::operator = (Tensor&& other) noexcept {
    this->operation = other.operation;
    swap(this->parents, other.parents);
    swap(this->output, other.output);
    swap(this->gradient, other.gradient);
    swap(this->childrenCount, other.childrenCount);
    swap(this->childrenGradReady, other.childrenGradReady);
    return *this;
};

const Blob& Tensor::forward() {
    if (!output) {
        vector<LazyBlobRef> datas;
        getParentsData(datas);
        // here the blob's move constructor is used
        this->output = operation.compute(datas);
        // Don't need references to parents datas anymore
        Allocator::endSession();
    }
    return *output;
}

void Tensor::backward(bool lastNode) {
    if (lastNode) {
        gradient = Blob::ones(1, 1);
    }
    // go backward only if we have parents
    if (!parents.size()) return;

    assert(childrenGradReady <= childrenCount);
    if (childrenGradReady < childrenCount) return;

    vector<LazyBlobRef> datas;
    getParentsData(datas);
    auto grads = operation.grad(*gradient, datas);
    assert(grads.size() == parents.size());
    for (int i = 0; i < parents.size(); ++i)
        parents[i].get().accumulate(grads[i]);

    // We need to end session in the allocator after all gradients
    // have been used and we don't need to store references to them anymore
    Allocator::endSession();

    for (auto p: parents) p.get().backward();
}

void Tensor::accumulate(const LazyBlob& gradient) {
    if (!this->gradient)
        this->gradient = gradient;
    else 
        *this->gradient += gradient;
    childrenGradReady++;
}

void Tensor::clear() {
    // if has parents, then clear output cache
    this->gradient = {};
    if (parents.size())
        this->output = {};
    this->childrenGradReady = 0;
    for (auto p: parents) p.get().clear();
}
