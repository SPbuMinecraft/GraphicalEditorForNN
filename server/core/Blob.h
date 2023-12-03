#pragma once

#include <iostream>
#include <cstddef>
#include <functional>

#include "LazyBlob.h"
#include "RandomInit.h"
#include "Shape.h"

typedef float (*UnaryTransform)(float x);

class Blob final {
private:
    float* data = NULL;
    Blob(Shape shape, const float* data, bool constMemory = false);
    Blob(Shape shape, bool constMemory, RandomObject* object = nullptr);

    float* getAddress(size_t k, size_t l, size_t i, size_t j) const;
public:
    /// Keep it `const`, PLEASE
    const Shape shape;

    Blob(Shape shape);
    Blob(float value);
    /// Move constructor, it takes data away from `other`
    Blob(Blob&& other) noexcept;
    /// Now it is ILLIGAL to copy a blob
    Blob(const Blob& other) = delete;
    /// But forcing evaluation in LazyBlob creates a Blob
    Blob(const LazyBlob& other);
    ~Blob();

    static Blob fill(Shape shape, float value);
    static Blob zeros(Shape shape);
    static Blob ones(Shape shape);
    static Blob constBlob(Shape shape, const float* data);
    static Blob constRandomBlob(Shape shape, RandomObject* object = nullptr);

    /// Convert to lazy with these
    operator const LazyBlob&() const;
    const LazyBlob& lazy() const;

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const;
    float operator() (std::size_t l, std::size_t i, std::size_t j) const;
    float operator() (std::size_t i, std::size_t j) const;
    float operator() (std::size_t j) const;

    float& operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j);
    float& operator() (std::size_t l, std::size_t i, std::size_t j);
    float& operator() (std::size_t i, std::size_t j);
    float& operator() (std::size_t j);

    void clear();

    /// Also move assignment, so `t` is useles after this
    Blob& operator = (Blob&& t) noexcept;

    friend bool operator == (const Blob &a, const Blob &b);
    friend bool operator != (const Blob &a, const Blob &b);
    friend std::ostream& operator<<(std::ostream& os, const LazyBlob& b);
};
