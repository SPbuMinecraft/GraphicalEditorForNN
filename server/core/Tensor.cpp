#include <algorithm>
#include <functional>
#include <cassert>

#include "Blob.h"
#include "Tensor.h"

using namespace std;

static const OpNone noop = OpNone();

void Tensor::getParentsData(vector<BlobRef> &datas) {
    for (auto p: parents) datas.push_back(p.get().forward());
}

void Tensor::getParentsGrads(vector<BlobRef> &grads) {
    for (auto p: parents) grads.push_back(p.get().gradient.value());
}

Tensor::Tensor(const Operation& operation, const vector<TensorRef>& parents, bool noGrad)
               : operation(operation), parents(parents) {
    if (!noGrad) {
        for (auto p: parents) p.get().childrenCount++;
    }

    vector<BlobRef> datas;
    getParentsData(datas);

    vector<size_t> dims = operation.computeDim(datas);
    output = Blob {dims[0], dims[1]};
    operation.compute(datas, output.value());
    outputIsNull = false;
    gradient = Blob {output->rows, output->cols};
}

Tensor::Tensor(const Blob& data): operation(noop), parents({}) {
    output = Blob {data.rows, data.cols, data.getData()};
    outputIsNull = false;
    gradient = Blob {data.rows, data.cols};
}

BlobRef Tensor::forward() {
    if (outputIsNull) {
        vector<BlobRef> datas;
        getParentsData(datas);
        operation.compute(datas, output.value());
        outputIsNull = false;
    }
    return *output;
}

void Tensor::backward() {
    vector<BlobRef> datas;
    vector<BlobRef> grads;
    getParentsData(datas);
    getParentsGrads(grads);
    operation.grad(*gradient, datas, grads);
    assert(grads.size() == parents.size());
    for (int i = 0; i < parents.size(); ++i)
        parents[i].get().accumulate();
}

void Tensor::accumulate(){
    childrenGradReady++;
    if (childrenGradReady == childrenCount)
        backward();
}

void Tensor::clear() {
    gradient->clear();
    // if has parents, then clear output cache
    if (parents.size()) {
        this->output->clear();
        outputIsNull = true;
    }
    this->childrenGradReady = 0;
    for (auto p: parents) p.get().clear();
}

Tensor Tensor::operator=(const Tensor & other) {
    this->operation = other.operation;
    this->outputIsNull = other.outputIsNull;
    this->childrenCount = other.childrenCount;
    this->childrenGradReady = other.childrenGradReady;
    this->parents = other.parents;
    this->output = other.output;
    this->gradient = other.gradient;
    return *this;
}
