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
    return a + b;
}
vector<LazyBlobRef> Sum::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    (void)args;
    return {grad, grad};
}
Shape Sum::computeDim(const std::vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {a.rows(), b.cols()};
}

Blob Multiply::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return a & b;
}
vector<LazyBlobRef> Multiply::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {grad & b.transposed(), a.transposed() & grad};
}
Shape Multiply::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {a.rows(), b.cols()};
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
    return {a.rows(), a.cols()};
}

Blob BiasSum::compute(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return a + b;
}
vector<LazyBlobRef> BiasSum::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args2(a, b);
    (void)a;

    Blob *bgrad = Allocator::allocateBlob(Shape {1, grad.shape.cols});

    for (int i = 0; i < grad.shape.rows; ++i)
        for (int j = 0; j < grad.shape.cols; ++j)
            *(bgrad->get_address(0, j)) = b(0, j) + grad(i, j);

    return {grad, *bgrad};
}
Shape BiasSum::computeDim(const vector<LazyBlobRef>& args) const {
    args2(a, b);
    return {a.rows(), b.cols()};
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
    return {a.rows(), a.cols()};
}

Blob Mean::compute(const vector<LazyBlobRef>& args) const {
    args1(a);
    return a.mean({2, 3});
}

vector<LazyBlobRef> Mean::grad(const Blob& grad, const vector<LazyBlobRef>& args) const {
    args1(a);
    return { grad / (a.rows() * a.cols()) };
}
Shape Mean::computeDim(const vector<LazyBlobRef>& args) const {
    (void)args;
    return {1, 1};
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
    return {a.rows(), a.cols()};
}
