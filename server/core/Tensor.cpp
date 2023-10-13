#include "Tensor.h"
#include <algorithm>
#include <functional>
#include <cassert>

using namespace std;

static const None noop = None();

void Tensor::getParentsData(vector<BlobRef> &datas) {
    for (auto p: parents) datas.push_back(p.get().forward());
}

Tensor::Tensor(const Operation& operation, const vector<TensorRef>& parents): operation(operation), parents(parents) {
    for (auto p: parents) p.get().childrenCount++;

    vector<BlobRef> datas;
    getParentsData(datas);

    Blob output = operation.compute(datas);
    this->output = output;
    Blob gradient {output.rows, output.cols};
    this->gradient = gradient;
}

Tensor::Tensor(const Blob& data): operation(noop), parents({}) {
    Blob output = data;
    this->output = output;
    Blob gradient {data.rows, data.cols};
    this->gradient = gradient;
}

BlobRef Tensor::forward() {
    if (!output) {
        vector<BlobRef> datas;
        getParentsData(datas);
        Blob output = operation.compute(datas);
        this->output = output;
    }
    return *output;
}

void Tensor::backward() {
    vector<BlobRef> datas;
    getParentsData(datas);
    vector<Blob> gs = operation.grad(*gradient, datas);
    assert(gs.size() == parents.size());
    for (int i = 0; i < parents.size(); ++i)
        parents[i].get().accumulate(gs[i]);
}

void Tensor::accumulate(const Blob& gradient){
    if (gradient.rows > this->gradient->rows) {
        for (int i = 0; i < gradient.rows; ++i) {
            for (int j = 0; j < gradient.cols; ++j) {
                (*this->gradient)[0][j] += gradient[i][j];
            }
        }
    } else {
        *this->gradient += gradient;
    }
    childrenGradReady++;
    if (childrenGradReady == childrenCount)
        backward();
}

void Tensor::clear() {
    gradient->clear();
    // if has parents, then clear output cache
    if (parents.size()) this->output = {};
    this->childrenGradReady = 0;
    for (auto p: parents) p.get().clear();
}
