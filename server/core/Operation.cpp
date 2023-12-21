#include <cassert>
#include <cmath>

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
    return {grad & b.transposed(), (a.transposed() & grad).sum({0})};
}
Shape Multiply::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {{a.shape().rows(), b.shape().cols()}};
}

Blob Divide::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    assert(a.shape() == b.shape());
    return a / (b + 1e-6);
}
vector<LazyBlobRef> Divide::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {grad / (b + 1e-6), grad * (-a / (b * b + 1e-6))};
}
Shape Divide::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    (void)b;
    return {a.shape()};
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
    return {a.shape()};
}

Blob Root::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.applying([](float x) { return std::sqrt(x); });
}

vector<LazyBlobRef> Root::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return {combine(a, grad, [](float x, float g) { return g / (2 * std::sqrt(x)); })};
}

Shape Root::computeDim(const std::vector<LazyBlobRef>& args) const {
    args1(a);
    return {a.shape()};
}

Mean::Mean(std::vector<short> axis): axis(axis) {}
Mean::Mean(std::vector<short> axis, bool minusOne): axis(axis) {
    this->minusOne = minusOne;
}

Blob Mean::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.mean(axis, minusOne);
}

vector<LazyBlobRef> Mean::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    size_t count = 1; 
    for (int dim: axis) {
        count *= a.shape()[dim];
    }
    if (minusOne) {
        count -= 1;
    }
    return { grad.lazy().fill(a.shape()) / count };
}
Shape Mean::computeDim(const vector<LazyBlobRef>& args) const {
    (void)args;
    return {{1, 1}};
}

SumAxis::SumAxis(std::vector<short> axis): axis(axis) {}

Blob SumAxis::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.sum(axis);
}

vector<LazyBlobRef> SumAxis::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return {grad.lazy().fill(a.shape())};
}
Shape SumAxis::computeDim(const vector<LazyBlobRef>& args) const {
    (void)args;
    return {{1, 1}};
}

Var::Var(std::vector<short> axis): axis(axis) {}

Blob Var::compute(const vector<LazyBlobRef>& args) const {
    size_t count = 1; 
    args1(a);
    for (int dim: axis) {
        count *= a.shape()[dim];
    }
    return (a - a.mean(axis).fill(a.shape())).sum(axis) * (a - a.mean(axis).fill(a.shape())).sum(axis) / count;
}

vector<LazyBlobRef> Var::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return { (grad / (a.shape().rows() * a.shape().cols())).fill(a.shape()) };
}
Shape Var::computeDim(const vector<LazyBlobRef>& args) const {
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

Blob Conv2D::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return conv(a, b);
}
vector<LazyBlobRef> Conv2D::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    Blob *bgrad = Allocator::allocateBlob(b.shape());
    for (int i = 0; i < a.shape().dim4(); ++i) {
        (*bgrad) += conv_i(a, grad.lazy(), b.shape().cols(), i);
    }
    return {
        conv(grad, b.transposeFirst2Dims().reverseLast2Dims()), 
        *bgrad
    };
}
Shape Conv2D::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {
        {
            a.shape().dim4(), 
            b.shape().dim4(), 
            a.shape().rows(), 
            a.shape().cols()
        }, 
        a.shape().dimsCount
    };
}

Blob Fill::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return b.fill(a.shape());
}

vector<LazyBlobRef> Fill::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    (void)a;
    std::vector<short> axisForSum = {};
    for (int i = 0; i < b.shape().getDims().size(); ++i) {
        if (b.shape().getDims()[i] != grad.shape.getDims()[i]) {
            axisForSum.push_back(i);
        }
    }
    return {zeroBlob(a.shape()), grad.lazy().sum(axisForSum)};
}

Shape Fill::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    (void)b;
    return a.shape();
}

Blob EPS::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a + eps;
}

vector<LazyBlobRef> EPS::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    (void)a;
    return {grad};
}

Shape EPS::computeDim(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.shape();
}

Blob Exp::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.applying([](float x) { return std::exp(x); });
}

vector<LazyBlobRef> Exp::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return {grad * a.applying([](float x) { return std::exp(x); })};
}

Shape Exp::computeDim(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.shape();
}

Blob Entropy::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return -a.entropy(b, classCount);
}

vector<LazyBlobRef> Entropy::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {grad.lazy().fill(a.shape()) * a.entropyDerivative(b, classCount), zeroBlob(b.shape())};
}

Shape Entropy::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    (void)a;
    return b.shape();
}

Blob MaxPoolOp::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.maxPool();
}

vector<LazyBlobRef> MaxPoolOp::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return {a.maxPoolDerivative(grad.lazy())};
}

Shape MaxPoolOp::computeDim(const vector<LazyBlobRef>& args) const {
    args1(a);
    return {{a.shape().dim4(), a.shape().dim3(), a.shape().rows() / 2, a.shape().cols() / 2}, a.shape().dimsCount};
}
