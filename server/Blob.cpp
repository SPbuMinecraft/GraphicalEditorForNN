#include "Blob.h"
#include <ostream>
#include <cassert>
#include <algorithm>

using namespace std;

Blob::Blob(size_t rows, size_t cols): rows(rows), cols(cols) {
    this->data = new float[rows * cols];
    memset(data, 0, rows * cols * sizeof(float));
}

Blob::Blob(size_t rows, size_t cols, float* data): rows(rows), cols(cols) {
    this->data = new float[rows * cols];
    copy_n(data, rows * cols, this->data);
}

Blob::Blob(const Blob& other): rows(other.rows), cols(other.cols) {
    data = new float[rows * cols];
    copy_n(other.data, rows * cols, data);
}

Blob::~Blob(){
    delete[] data;
}

float Blob::at(size_t i, size_t j) const {
    return (*this)[i][j];
}

const float* Blob::operator[](size_t index) const {
    const float* tmp = (data + cols * index);
    return tmp;
}

float* Blob::operator[](size_t index) {
    return (data + cols * index);
}

bool Blob::operator==(const Blob& b) const {
    if (rows != b.rows || cols != b.cols) return false;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if ((*this)[i][j] != b[i][j]) return false;
    return true;
}

bool Blob::operator!=(const Blob& b) const {
    return !(*this == b);
}

Blob Blob::operator-() const {
    Blob tmp = *this;
    transform(tmp.data, tmp.data + (rows * cols), tmp.data, [](float x){ return -x; });
    return tmp;
}

Blob Blob::operator+(const Blob& t) const {
    Blob tmp = *this;
    tmp += t;
    return tmp;
}

Blob Blob::operator-(const Blob& t) const {
    Blob tmp = *this;
    tmp -= t;
    return tmp;
}

Blob Blob::operator*(const Blob& t) const {
    assert(cols == t.rows);
    Blob result {rows, t.cols};
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < t.cols; ++j)
            for (int k = 0; k < cols; ++k)
                result[i][j] += (*this)[i][k] * t[k][j];
    return result;
}

Blob Blob::transposed() const {
    Blob result {cols, rows};
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[j][i] = (*this)[i][j];
    return result;
}

void swap(Blob& first, Blob& second) {
    assert(first.rows == second.rows && first.cols == second.cols);
    swap(first.data, second.data);
}

Blob& Blob::operator=(const Blob& t) {
    Blob tmp(t);
    swap(*this, tmp);
    return *this;
}

Blob& Blob::operator+=(const Blob& t) {
    assert(rows == t.rows && cols == t.cols);
    transform(data, data + (rows * cols), t.data, data, [](float x, float y){ return (x + y); });
    return *this;
}

Blob& Blob::operator-=(const Blob& t) {
    assert(rows == t.rows && cols == t.cols);
    transform(data, data + (rows * cols), t.data, data, [](float x, float y){ return (x - y); });
    return *this;
}

ostream& operator<<(ostream& os, const Blob& b) {
    for (int i = 0; i < b.rows; ++i) {
        for (int j = 0; j < b.cols; ++j)
            os << b[i][j] << " ";
        os << endl;
    }
    return os;
}
