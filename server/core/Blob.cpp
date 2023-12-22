#include "Blob.h"
#include "Allocator.h"
#include "RandomInit.h"
#include <cassert>
#include <algorithm>
#include <cstring>

using namespace std;

Blob::Blob(Shape shape, bool constMemory, RandomObject* object): shape(shape) {
    this->data = Allocator::allocate(shape, constMemory);
    if (object == nullptr) clear();
    else object->simpleInit(data, shape.size());
}

Blob::Blob(Shape shape, const float* data, bool constMemory): Blob(shape, constMemory) {
    copy_n(data, shape.size(), this->data);
}

Blob::Blob(Shape shape): Blob(shape, false) {}

Blob::Blob(float value): Blob(Shape(), &value) {};

Blob::Blob(Blob&& other) noexcept: shape(other.shape) {
    *this = std::move(other);
}

Blob::Blob(const LazyBlob& other): Blob(Shape {other.shape()}) {
    for (int k = 0; k < shape.dim4(); ++k)
        for (int l = 0; l < shape.dim3(); ++l)
            for (int i = 0; i < shape.rows(); ++i)
                for (int j = 0; j < shape.cols(); ++j)
                    (*this)(k, l, i, j) = other(k, l, i, j);
}

Blob::~Blob() {
    if (data) Allocator::release(data);
}

Blob::operator const LazyBlob&() const {
    return lazy();
}
const LazyBlob& Blob::lazy() const {
    void *location = Allocator::allocateBytes(sizeof(LazyBlobView));
    return *(new(location) LazyBlobView(*this));
}

float Blob::operator() (size_t k, size_t l, size_t i, size_t j) const {
    assert(shape.dimsCount > 0);
    return *getAddress(k, l, i, j);
}

float Blob::operator() (size_t l, size_t i, size_t j) const {
    assert(shape.dimsCount == 3);
    return *getAddress(0, l, i, j);
}

float Blob::operator() (size_t i, size_t j) const {
    assert(shape.dimsCount == 2);
    return *getAddress(0, 0, i, j);
}

float Blob::operator() (size_t j) const {
    assert(shape.dimsCount == 1);
    return *getAddress(0, 0, 0, j);
}

float& Blob::operator() (size_t k, size_t l, size_t i, size_t j) {
    assert(shape.dimsCount > 0);
    return *getAddress(k, l, i, j);
}

float& Blob::operator() (size_t l, size_t i, size_t j) {
    assert(shape.dimsCount == 3);
    return *getAddress(0, l, i, j);
}

float& Blob::operator() (size_t i, size_t j) {
    assert(shape.dimsCount == 2);
    return *getAddress(0, 0, i, j);
}

float& Blob::operator() (size_t j) {
    assert(shape.dimsCount == 1);
    return *getAddress(0, 0, 0, j);
}

float* Blob::getAddress(size_t k, size_t l, size_t i, size_t j) const {
    const size_t colStride = shape.cols();
    const size_t rowStride = shape.rows() * colStride;
    const size_t dim3Stride = shape.dim3() * rowStride;
    return data + k * dim3Stride + l * rowStride + i * colStride + j;
//    size_t indices[] = {k, l, i, j};
//    
//    size_t res = 0;
//    for (short int i = 0; i < 4; ++i)
//        res += shape.stride(i) * indices[i];
//    return data + res;
}

void Blob::clear() {
    memset(this->data, 0, shape.size() * sizeof(float));
}

Blob& Blob::operator = (Blob&& t) noexcept {
    assert(shape == t.shape);
    this->data = exchange(t.data, nullptr);
    return *this;
}

bool operator == (const Blob &a, const Blob &b) {
    if (!(a.shape == b.shape)) return false;
    return equal(a.data, a.data + a.shape.size(), b.data);
};
bool operator != (const LazyBlob &a, const LazyBlob &b) {
    return !(a == b);
};

std::ostream& operator<<(std::ostream& os, const Blob& b) {
    for (int l = 0; l < b.shape.dim4(); ++l) {
        for (int k = 0; k < b.shape.dim3(); ++k){
            for (int i = 0; i < b.shape.rows(); ++i) {
                for (int j = 0; j < b.shape.cols(); ++j)
                    os << b(l, k, i, j) << " ";
                os << std::endl;
            }
            os << std::endl;
        }
        os << std::endl;
    }
    return os;
};

Blob Blob::fill(Shape shape, float value) {
    Blob a {shape};
    fill_n(a.data, shape.size(), value);
    return a;
}
Blob Blob::zeros(Shape shape) {
    return fill(shape, 0);
}
Blob Blob::ones(Shape shape) {
    return fill(shape, 1);
}

Blob Blob::constBlob(Shape shape, const float* data) {
    return Blob(shape, data, true);
}

Blob Blob::constRandomBlob(Shape shape, RandomObject* object) {
    return Blob(shape, true, object);
}
