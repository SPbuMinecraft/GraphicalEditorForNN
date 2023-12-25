#include <cstddef>
#include <string>
#include <vector>
#include <cassert>

#include "Shape.h"

using namespace std;

Shape::Shape(vector<size_t> dims): Shape(dims, dims.size()) {}

Shape::Shape(vector<size_t> dims, short dimsCount): dimsCount(dimsCount) {
    assert(dims.size() <= 4);
    for (; dims.size() < 4; dims.insert(dims.begin(), 1));
    for (int i = 0; i < 4; i++)
        this->dims[i] = dims[i];
}

Shape::Shape(const Shape& other): dimsCount(other.dimsCount) {
    for (int i = 0; i < 4; i++) {
        this->dims[i] = other.dims[i];
    }
}

Shape& Shape::operator = (const Shape& other) {
    this->dimsCount = other.dimsCount;
    for (int i = 0; i < 4; i++) {
        this->dims[i] = other.dims[i];
    }
    return *this;
}

bool Shape::operator == (const Shape& other) const {
    return cols() == other.cols() && rows() == other.rows() && dim3() == other.dim3() && dim4() == other.dim4();
}

bool Shape::operator != (const Shape& other) const {
    return !(*this == other);
}

size_t Shape::size() const {
    return cols() * rows() * dim3() * dim4();
}

string Shape::toString() const {
    return to_string(dim4()) + "x" + to_string(dim3()) + "x" + to_string(rows()) + "x" + to_string(cols());
}

vector<size_t> Shape::getDims() const {
    vector<size_t> dims;
    for (int i = 0; i < 4; i++) {
        dims.push_back(this->dims[i]);
    }
    return dims;
}

size_t& Shape::operator [] (int i) {
    return dims[i];
}
    
size_t Shape::operator [] (int i) const {
    return dims[i];
}

size_t Shape::stride(int i) const {
    assert(i >= 0 && i < 4);
    switch (i) {
        case 0: return dims[1] * dims[2] * dims[3];
        case 1: return dims[2] * dims[3];
        case 2: return dims[3];
        default: return 1;
    }
}

size_t Shape::cols() const {
    return dims[3];
}

size_t Shape::rows() const {
    return dims[2];
}

size_t Shape::dim3() const {
    return dims[1];
}

size_t Shape::dim4() const {
    return dims[0];
}

inline void hash_combine(std::size_t& seed, size_t v) {
    seed ^= v + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

size_t hash<Shape>::operator()(const Shape& shape) const {
    size_t seed = 1843;

    for (int i = 0; i < 4; i++) {
        hash_combine(seed, hash<size_t>()(shape[i]));
    }

    return seed;
};
