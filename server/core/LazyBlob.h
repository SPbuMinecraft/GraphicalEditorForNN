#pragma once

#include <iostream>
#include <vector>

#include "Shape.h"

typedef float (*UnaryTransform)(float x);
typedef float (*BinaryTransform)(float x, float y);

class LazyBlob;
typedef std::reference_wrapper<const LazyBlob> LazyBlobRef;

class Blob;
class LazyBlob {
public:
    std::size_t rows() const { return shape().rows; };
    std::size_t cols() const { return shape().cols; };
    virtual Shape shape() const = 0;
    virtual float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const = 0;

    float operator() (std::size_t l, std::size_t i, std::size_t j) const { return (*this)(0, l, i, j); };
    float operator() (std::size_t i, std::size_t j) const { return (*this)(0, 0, i, j); };
    float operator() (std::size_t j) const { return (*this)(0, 0, 0, j); };

    const LazyBlob& dot(const LazyBlob& b) const;
    const LazyBlob& transposed() const;
    const LazyBlob& applying(const UnaryTransform t) const;
    const LazyBlob& sum(std::vector<std::size_t> axis) const;
    const LazyBlob& mean(std::vector<std::size_t> axis) const;


    friend const LazyBlob& operator + (const LazyBlob &a, const LazyBlob &b);
    friend const LazyBlob& operator - (const LazyBlob &a, const LazyBlob &b);
    friend const LazyBlob& operator - (const LazyBlob &a);
    /// ELEMENT-WISE
    friend const LazyBlob& operator * (const LazyBlob &a, const LazyBlob &b);
    /// MATRIX
    friend const LazyBlob& operator & (const LazyBlob &a, const LazyBlob &b);

    /// Inplace operations, you can also pass another blob as second argument
    friend Blob& operator += (Blob& a, const LazyBlob& b);
    friend Blob& operator -= (Blob& a, const LazyBlob& b);
    friend Blob& operator *= (Blob& a, const LazyBlob& b);

    friend std::ostream& operator<<(std::ostream& os, const LazyBlob& b);
};

class LazyBlobView final: public LazyBlob {
private:
    const Blob &ref;
public:
    LazyBlobView(const Blob &ref);

    Shape shape() const override;

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override;
};

const LazyBlob& operator + (float a, const LazyBlob &b);
const LazyBlob& operator - (float a, const LazyBlob &b);
const LazyBlob& operator * (float a, const LazyBlob &b);

const LazyBlob& operator + (const LazyBlob &a, float b);
const LazyBlob& operator - (const LazyBlob &a, float b);
const LazyBlob& operator * (const LazyBlob &a, float b);
const LazyBlob& operator / (const LazyBlob &a, float b);

const LazyBlob& combine(const LazyBlob &a, const LazyBlob &b, const BinaryTransform how);
