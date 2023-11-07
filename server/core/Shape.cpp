#include <cstddef>
#include <string>

#include "Shape.h"

using namespace std;

Shape::Shape(): dim4(1), dim3(1), rows(1), cols(1), dimsCount(0) { fillDims(); }

Shape::Shape(size_t cols): dim4(1), dim3(1), rows(1), cols(cols), dimsCount(1) { fillDims(); }

Shape::Shape(size_t rows, size_t cols): dim4(1), dim3(1), rows(rows), cols(cols), dimsCount(2) { fillDims(); }

Shape::Shape(size_t dim3, size_t rows, size_t cols): dim4(1), dim3(dim3), rows(rows), cols(cols), dimsCount(3) { fillDims(); }

Shape::Shape(size_t dim4, size_t dim3, size_t rows, size_t cols): dim4(dim4), dim3(dim3), rows(rows), cols(cols), dimsCount(4) { fillDims(); }

Shape::Shape(size_t dim4, size_t dim3, size_t rows, size_t cols, short int dimsCount): 
    dim4(dim4), dim3(dim3), rows(rows), cols(cols), dimsCount(dimsCount) { fillDims(); }

bool Shape::operator == (const Shape& other) const {
    return this->cols == other.cols && this->rows == other.rows && this->dim3 == other.dim3 && this->dim4 == other.dim4;
}

bool Shape::operator != (const Shape& other) const {
    return !(*this == other);
}

size_t Shape::size() const {
    return cols * rows * dim3 * dim4;
}

string Shape::toString() const {
    return to_string(cols) + "x" + to_string(rows) + "x" + to_string(dim3) + "x" + to_string(dim4);
}

void Shape::fillDims() {
    dims[0] = dim4;
    dims[1] = dim3;
    dims[2] = rows;
    dims[3] = cols;

    dimsMult[3] = 1;
    dimsMult[2] = cols * dimsMult[3];
    dimsMult[1] = rows * dimsMult[2];
    dimsMult[0] = dim3 * dimsMult[1];
}
