#pragma once

#include <iostream>
#include <cstddef>

typedef float (*UnaryTransform)(float x);

class Blob {
    float* data;
public:
    const std::size_t rows;
    const std::size_t cols;

    Blob(std::size_t rows, std::size_t cols, const float* data);
    Blob(std::size_t rows, std::size_t cols);
    Blob();
    Blob(const Blob& other);
    ~Blob();

    float at(std::size_t i, std::size_t j) const;
    const float* operator[](std::size_t index) const;
    float* operator[](std::size_t index);

    void clear();

    bool operator==(const Blob& t) const;
    bool operator!=(const Blob& t) const;

    Blob operator-() const;
    Blob operator+(const Blob& t) const;
    Blob operator-(const Blob& t) const;
    Blob operator*(const Blob& t) const;
    Blob transposed() const;
    Blob applying(UnaryTransform transform) const;

    friend Blob operator*(float x, const Blob& b);
    friend Blob operator*(const Blob& b, float x);

    friend void swap(Blob& first, Blob& second);
    Blob& operator=(const Blob& t);

    Blob& operator+=(const Blob& t);
    Blob& operator-=(const Blob& t);
    Blob& operator*=(const Blob& t);

    friend std::ostream& operator<<(std::ostream& os, const Blob& b);
};

typedef std::reference_wrapper<Blob> BlobRef;
