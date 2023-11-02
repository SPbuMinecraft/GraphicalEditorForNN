#include "Blob.h"
#include "Allocator.h"
#include "RandomInit.h"
#include <cassert>
#include <algorithm>
#include <cstring>

using namespace std;

Blob::Blob(size_t rows, size_t cols, bool constMemory, RandomObject* object): rows(rows), cols(cols) {
    this->data = Allocator::allocate({rows, cols}, constMemory);
    if (object == nullptr) clear();
    else object->simpleInit(data, rows * cols);
}

Blob::Blob(size_t rows, size_t cols, const float* data, bool constMemory): Blob(rows, cols, constMemory) {
    copy_n(data, rows * cols, this->data);
}

Blob::Blob(size_t rows, size_t cols): Blob(rows, cols, false) {}

Blob::Blob(float value): Blob(1, 1, &value) {};

Blob::Blob(Blob&& other) noexcept: rows(other.rows), cols(other.cols) {
    *this = std::move(other);
}

Blob::Blob(const LazyBlob& other): Blob(other.rows(), other.cols()) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            (*this)[i][j] = other(i, j);
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

float Blob::operator() (size_t i, size_t j) const {
    return (*this)[i][j];
}
const float* Blob::operator [] (size_t index) const {
    return data + cols * index;
}
float* Blob::operator [] (size_t index) {
    return data + cols * index;
}

void Blob::clear() {
    memset(this->data, 0, rows * cols * sizeof(float));
}

Blob& Blob::operator = (Blob&& t) noexcept {
    assert(rows == t.rows && cols == t.cols);
    this->data = exchange(t.data, nullptr);
    return *this;
}

bool operator == (const Blob &a, const Blob &b) {
    if (a.rows!= b.rows || a.cols!= b.cols) return false;
    return equal(a.data, a.data + a.rows * a.cols, b.data);
};
bool operator != (const LazyBlob &a, const LazyBlob &b) {
    return !(a == b);
};

std::ostream& operator<<(std::ostream& os, const Blob& b) {
    for (int i = 0; i < b.rows; ++i) {
        for (int j = 0; j < b.cols; ++j)
            os << b(i, j) << " ";
        os << std::endl;
    }
    return os;
};

Blob Blob::fill(size_t rows, size_t cols, float value) {
    Blob a {rows, cols};
    fill_n(a.data, rows * cols, value);
    return a;
}
Blob Blob::zeros(size_t rows, size_t cols) {
    return fill(rows, cols, 0);
}
Blob Blob::ones(size_t rows, size_t cols) {
    return fill(rows, cols, 1);
}

Blob Blob::constBlob(std::size_t rows, std::size_t cols, const float* data) {
    return Blob(rows, cols, data, true);
}

Blob Blob::constBlobRandom(std::size_t rows, std::size_t cols, RandomObject* object) {
    return Blob(rows, cols, true, object);
}
