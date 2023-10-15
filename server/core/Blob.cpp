#include <cassert>
#include <algorithm>
#include <cstring>

#include "Blob.h"
#include "RandomInit.h"

using namespace std;

Blob::Blob(size_t rows, size_t cols, const float* data): rows(rows), cols(cols) {
    this->data = new float[rows * cols];
    copy_n(data, rows * cols, this->data);
}

Blob::Blob(size_t rows, size_t cols, const float value): rows(rows), cols(cols) {
    this->data = new float[rows * cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i * cols + j] = value;
}

Blob::Blob(size_t rows, size_t cols, RandomObject* object): rows(rows), cols(cols) {
    this->data = new float[rows * cols];
    if (object == nullptr) clear();
    else object->simpleInit(this->data, rows * cols);
}

Blob::Blob(const Blob& other): Blob(other.rows, other.cols, other.data) {}

Blob::Blob(): data(nullptr), rows(0), cols(0) {}

Blob::~Blob() {
    delete[] data;
}

float* Blob::getData() const {
    return data;
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

void Blob::clear() {
    memset(this->data, 0, rows * cols * sizeof(float));
}

bool Blob::operator==(const Blob& b) const {
    if (rows != b.rows || cols != b.cols) return false;
    return equal(this->data, this->data + rows * cols, b.data);
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
    const Blob *big = this;
    const Blob *small = &t;

    if (big->rows < small->rows) swap(big, small);

    Blob tmp = *big;
    tmp += *small;
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

Blob Blob::applying(UnaryTransform transform) const {
    Blob result {rows, cols};
    std::transform(data, data + rows * cols, result.data, transform);
    return result;
}

Blob operator*(float x, const Blob& b) {
    Blob result {b.rows, b.cols};
    transform(b.data, b.data + b.rows * b.cols, result.data,
              [x](float v) { return x * v; });
    return result;
}

Blob operator*(const Blob& b, float x) {
    return x * b;
}

void swap(Blob& first, Blob& second) {
    assert(first.rows == second.rows && first.cols == second.cols);
    swap(first.data, second.data);
}

Blob& Blob::operator=(const Blob& t) {
    if (this == &t) return *this;

    memcpy(this->data, t.data, sizeof(float) * t.cols * t.rows);
    return *this;
}

Blob& Blob::operator+=(const Blob& t) {
    assert(rows % t.rows == 0);
    assert(cols % t.cols == 0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; c += t.cols)
            transform(t.data + (r % t.rows) * t.cols,
                      t.data + (r % t.rows + 1) * t.cols,
                      data + r * cols + c,
                      data + r * cols + c,
                      [](float x, float y) { return x + y; });
    return *this;
}

Blob& Blob::operator*=(const float t) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            data[r * cols + c] *= t;
            
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
