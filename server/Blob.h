#pragma once

#include <ostream>
#include <cstddef>

class Blob {
    float* data;
    Blob(std::size_t rows, std::size_t cols);
public:
    const std::size_t rows;
    const std::size_t cols;

    Blob(std::size_t rows, std::size_t cols, float* data);
    Blob(const Blob& other);
    ~Blob();

    float at(std::size_t i, std::size_t j) const;
    const float* operator[](std::size_t index) const;
    float* operator[](std::size_t index);

    bool operator==(const Blob& t) const;
    bool operator!=(const Blob& t) const;

    Blob operator-() const;
    Blob operator+(const Blob& t) const;
    Blob operator-(const Blob& t) const;
    Blob operator*(const Blob& t) const;
    Blob transposed() const;

    friend void swap(Blob& first, Blob& second);
    Blob& operator=(const Blob& t);

    Blob& operator+=(const Blob& t);
    Blob& operator-=(const Blob& t);
    Blob& operator*=(const Blob& t);

    friend std::ostream& operator<<(std::ostream& os, const Blob& b);
};
