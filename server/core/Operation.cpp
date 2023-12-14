#include <cassert>

#include "Allocator.h"
#include "Operation.h"

using namespace std;

const Operation& Operation::operator=(const Operation& other) const {
    return other;
}

#define args1(a) auto &a = args[0].get()
#define args2(a, b) auto &a = args[0].get(); auto &b = args[1].get()

Blob Noop::compute(const vector<LazyBlobRef>& args) const {
    (void)args;
    throw runtime_error("Unreachable");
}
vector<LazyBlobRef> Noop::grad(const Blob& gradient, const vector<LazyBlobRef>& args) const {
    (void)gradient;
    (void)args;
    throw runtime_error("Unreachable");
}
Shape Noop::computeDim(const vector<LazyBlobRef>& args) const {
    (void)args;
    throw runtime_error("Unreachable");
}

Blob Sum::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    assert(a.shape() == b.shape());
    return a + b;
}
vector<LazyBlobRef> Sum::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    (void)args;
    return {grad, grad};
}
Shape Sum::computeDim(const std::vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {{a.shape().rows(), b.shape().cols()}};
}

Blob Multiply::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    assert(a.shape().cols() == b.shape().rows());
    return a & b;
}
vector<LazyBlobRef> Multiply::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {grad & b.transposed(), a.transposed() & grad};
}
Shape Multiply::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {{a.shape().rows(), b.shape().cols()}};
}

Blob ReLU::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.applying([](float x) { return x >= 0 ? x : 0; });
}
vector<LazyBlobRef> ReLU::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return {combine(a, grad, [](float x, float g) { return x >= 0 ? g : 0; })};
}
Shape ReLU::computeDim(const vector<LazyBlobRef>& args) const {
    args1(a);
    return {{a.shape().rows(), a.shape().cols()}};
}

Blob BiasSum::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    if (!stretch.has_value()) {
        auto [canStretch, theStretch] = Stretch::canStretch(a.shape(), b.shape());
        assert(canStretch);
        stretch = new Stretch(theStretch);
    }
    return a + b;
}
vector<LazyBlobRef> BiasSum::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    assert(stretch.has_value());
    return {grad, grad.lazy().sum(stretch.value()->axisForStretch)};
}
Shape BiasSum::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {{a.shape().rows(), b.shape().cols()}};
}

Blob Square::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a * a;
}
vector<LazyBlobRef> Square::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return {2 * grad * a};
}
Shape Square::computeDim(const std::vector<LazyBlobRef>& args) const {
    args1(a);
    return {{a.shape().rows(), a.shape().cols()}};
}

Mean::Mean(std::vector<short> axis): axis(axis) {}

Blob Mean::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.mean(axis);
}

vector<LazyBlobRef> Mean::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return { (grad / (a.shape().rows() * a.shape().cols())).fill(a.shape()) };
}
Shape Mean::computeDim(const vector<LazyBlobRef>& args) const {
    (void)args;
    return {{1, 1}};
}

Blob Substract::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return a - b;
}
vector<LazyBlobRef> Substract::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    (void)args;
    return {grad, -grad.lazy()};
}
Shape Substract::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    (void)b;
    return {{a.shape().rows(), a.shape().cols()}};
}
