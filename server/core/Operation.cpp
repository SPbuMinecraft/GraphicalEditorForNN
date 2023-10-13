#include "Operation.h"

using namespace std;

Blob None::compute(const vector<BlobRef>& args) const {
    return {};
}
vector<Blob> None::grad(Blob& gradient, const vector<BlobRef>& args) const {
    return {};
}

Blob Sum::compute(const vector<BlobRef>& args) const {
    return args[0].get() + args[1].get();
}
vector<Blob> Sum::grad(Blob& grad, const vector<BlobRef>& args) const {
    return {grad, grad};
}

Blob Multiply::compute(const vector<BlobRef>& args) const {
    return args[0].get() * args[1].get();
}
vector<Blob> Multiply::grad(Blob& grad, const vector<BlobRef>& args) const {
    Blob ga = grad * args[1].get().transposed();
    Blob gb = args[0].get().transposed() * grad;
    return {ga, gb};
}

Blob Loss::compute(const vector<BlobRef>& args) const {
    Blob l = args[0].get() - args[1].get();
    return l.applying([](float x) { return x * x; });
}
vector<Blob> Loss::grad(Blob& grad, const vector<BlobRef>& args) const {
    BlobRef a = args[0];
    BlobRef b = args[1];
    Blob ga = 2 * (a.get() - b.get());
    Blob gb = 2 * (b.get() - a.get());
    return {ga, gb};
}
