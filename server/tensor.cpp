#include "tensor.hpp"
#include <algorithm>


tensor::tensor(size_t rows, size_t cols, float* data) {
    this->rows = rows;
    this->cols = cols;
    this->data = new float[rows * cols];
    std::copy_n(data, rows * cols, this->data);
}

tensor::tensor(const tensor& other)
{
    rows = other.rows;
    cols = other.cols;
    grad = nullptr;
    data = new float[rows * cols];
    std::copy_n(other.data, rows * cols, data);
}

tensor::~tensor(){
    delete[] data;
    //delete[] grad; ?
};



const float* tensor::operator[](std::size_t index) const {
    const float* tmp = (data + cols * index);
    return tmp;
};

float* tensor::operator[](std::size_t index) {
    return (data + cols * index);
};



tensor tensor::operator-(){
    tensor tmp = *this;
    std::transform(tmp.data, tmp.data + (rows * cols), tmp.data, [](float x){ return -x; });
    return tmp;
};

tensor tensor::operator+(const tensor& t) const {
    tensor tmp = *this;
    tmp += t;
    return tmp;
};

tensor tensor::operator-(const tensor& t) const {
    tensor tmp = *this;
    tmp -= t;
    return tmp;
};

//tensor tensor::operator*(const tensor& t) const {};



void swap(tensor& first, tensor& second) {
    std::swap(first.data, second.data);
    std::swap(first.cols, second.cols);
    std::swap(first.rows, second.rows);
    std::swap(first.grad, second.grad);
};

tensor& tensor::operator=(const tensor& t) {
    tensor tmp(t);
    swap(*this, tmp);
    return *this;
};

tensor& tensor::operator+=(const tensor& t) {
    std::transform(data, data + (rows * cols), t.data, data, [](float x, float y){ return (x + y); });
    return *this;
};

tensor& tensor::operator-=(const tensor& t) {
    std::transform(data, data + (rows * cols), t.data, data, [](float x, float y){ return (x - y); });
    return *this;
};

//tensor& tensor::operator*=(const tensor& t) {};