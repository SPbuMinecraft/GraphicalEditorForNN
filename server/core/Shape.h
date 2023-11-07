#pragma once

#include <cstddef>
#include <string>

struct Shape {
    std::size_t dim4, dim3, rows, cols;
    short int dimsCount;
    std::size_t dimsMult[4];
    std::size_t dims[4];
    Shape();
    Shape(std::size_t cols);
    Shape(std::size_t rows, std::size_t cols);
    Shape(std::size_t dim3, std::size_t rows, std::size_t cols);
    Shape(std::size_t dim4, std::size_t dim3, std::size_t rows, std::size_t cols);
    Shape(std::size_t dim4, std::size_t dim3, std::size_t rows, std::size_t cols, short int dimsCount);

    void fillDims();
    bool operator == (const Shape& other) const;
    bool operator != (const Shape& other) const;

    std::size_t size() const;
    std::string toString() const;
};