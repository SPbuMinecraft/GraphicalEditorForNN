#pragma once

#include <iostream>
#include <vector>
#include <optional>

#include "Shape.h"

typedef float (*UnaryTransform)(float x);
typedef float (*BinaryTransform)(float x, float y);

class LazyBlob;
typedef std::reference_wrapper<const LazyBlob> LazyBlobRef;

class Blob;
class LazyBlob {
public:
    const Shape& shape() const;
    mutable std::optional<Shape> shape_ = {};
    virtual void initShape() const = 0;
    virtual float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const = 0;

    float operator() (std::size_t l, std::size_t i, std::size_t j) const { return (*this)(0, l, i, j); };
    float operator() (std::size_t i, std::size_t j) const { return (*this)(0, 0, i, j); };
    float operator() (std::size_t j) const { return (*this)(0, 0, 0, j); };

    const LazyBlob& dot(const LazyBlob& b) const;
    const LazyBlob& transposed() const;
    const LazyBlob& applying(const UnaryTransform t) const;
    const LazyBlob& sum(std::vector<short> axis) const;
    const LazyBlob& mean(std::vector<short> axis, bool minusOne = false) const;
    const LazyBlob& reverseLast2Dims() const;
    const LazyBlob& transposeFirst2Dims() const;
    const LazyBlob& entropy(const LazyBlob& b, int classCount) const;
    const LazyBlob& entropyDerivative(const LazyBlob& b, int classCount) const;
    const LazyBlob& maxPool() const;
    const LazyBlob& maxPoolDerivative(const LazyBlob& b) const;

    /// To repeat some dimensions several times
    /// - Parameter shape: the size we want to get
    const LazyBlob& fill(Shape shape) const;

    friend const LazyBlob& operator + (const LazyBlob &a, const LazyBlob &b);
    friend const LazyBlob& operator - (const LazyBlob &a, const LazyBlob &b);
    friend const LazyBlob& operator - (const LazyBlob &a);
    /// ELEMENT-WISE
    friend const LazyBlob& operator * (const LazyBlob &a, const LazyBlob &b);
    friend const LazyBlob& operator / (const LazyBlob &a, const LazyBlob &b);
    /// MATRIX
    friend const LazyBlob& operator & (const LazyBlob &a, const LazyBlob &b);

    /// Inplace operations, you can also pass another blob as second argument
    friend Blob& operator += (Blob& a, const LazyBlob& b);
    friend Blob& operator -= (Blob& a, const LazyBlob& b);
    friend Blob& operator *= (Blob& a, const LazyBlob& b);

    friend std::ostream& operator<<(std::ostream& os, const LazyBlob& b);
};

LazyBlob&  conv(const LazyBlob &a, const LazyBlob &b);
LazyBlob&  conv_i(const LazyBlob &a, const LazyBlob &b, std::size_t kernelSize, std::size_t i);
LazyBlob& zeroBlob(const Shape& shape);

class LazyBlobView final: public LazyBlob {
private:
    const Blob &ref;
public:
    LazyBlobView(const Blob &ref);

    void initShape() const override;

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

// need for debug
void printBlob(const LazyBlob& a);
