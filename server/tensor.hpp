#ifndef TENSOR
#define TENSOR

#include <cstddef>

class tensor
{
private:
    size_t rows;
    size_t cols;
    float* data;
public:
    float* grad = nullptr;

public:
    tensor(size_t rows, size_t cols, float* data);
    tensor(const tensor& other);
    ~tensor();

    const float* operator[](std::size_t index) const;
    float* operator[](std::size_t index);

    tensor operator-();
    tensor operator+(const tensor& t) const;
    tensor operator-(const tensor& t) const;
    //tensor operator*(const tensor& t) const;

    friend void swap(tensor& first, tensor& second);
    tensor& operator=(const tensor& t);

    tensor& operator+=(const tensor& t);
    tensor& operator-=(const tensor& t);
    //tensor& operator*=(const tensor& t);

};


#endif