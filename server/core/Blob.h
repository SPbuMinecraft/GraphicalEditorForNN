#pragma once

#include "LazyBlob.h"
#include "RandomInit.h"

#include <iostream>
#include <cstddef>
#include <functional>

typedef float (*UnaryTransform)(float x);

class Blob final {
private:
    float* data = NULL;
public:
    /// Keep it `const`, PLEASE
    const std::size_t rows;
    const std::size_t cols;

    Blob(std::size_t rows, std::size_t cols, const float* data);
    Blob(std::size_t rows, std::size_t cols, RandomObject* object = nullptr);
    Blob(float value);
    /// Move constructor, it takes data away from `other`
    Blob(Blob&& other) noexcept;
    /// Now it is ILLIGAL to copy a blob
    Blob(const Blob& other) = delete;
    /// But forcing evaluation in LazyBlob creates a Blob
    Blob(const LazyBlob& other);
    ~Blob();

    static Blob fill(std::size_t rows, std::size_t cols, float value);
    static Blob zeros(std::size_t rows, std::size_t cols);
    static Blob ones(std::size_t rows, std::size_t cols);

    /// Convert to lazy with these
    operator const LazyBlob&() const;
    const LazyBlob& lazy() const;

    float operator() (std::size_t i, std::size_t j) const;
    const float* operator[] (std::size_t index) const;
    float* operator[] (std::size_t index);

    void clear();

    /// Also move assignment, so `t` is useles after this
    Blob& operator = (Blob&& t) noexcept;

    friend bool operator == (const Blob &a, const Blob &b);
    friend bool operator != (const Blob &a, const Blob &b);
    friend std::ostream& operator<<(std::ostream& os, const LazyBlob& b);
};
