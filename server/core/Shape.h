#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct Shape {
    short dimsCount;
    size_t strides[3];

    Shape(std::vector<size_t> dims = {});
    Shape(std::vector<size_t> dims, short dimsCount);

    Shape(const Shape& shape);
    Shape& operator = (const Shape& shape);

    bool operator == (const Shape& other) const;
    bool operator != (const Shape& other) const;
    size_t& operator [] (int i);
    size_t operator [] (int i) const;

    std::size_t size() const;
    std::string toString() const;
    std::vector<std::size_t> getDims() const;
    void calculateStrides();

    std::size_t cols() const;
    std::size_t rows() const;
    std::size_t dim3() const;
    std::size_t dim4() const;

    std::size_t stride(int i) const;

private:
    std::size_t dims[4];
    std::size_t cachedSize;
};

template<>
struct std::hash<Shape> {
    std::size_t operator()(const Shape& k) const;
};
