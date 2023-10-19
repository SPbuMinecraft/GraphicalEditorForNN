#include "Operation.h"

using namespace std;

const Operation& Operation::operator=(const Operation& other) const {
    return other;
}

void OpNone::compute(const vector<BlobRef>& args, Blob& res) const {
}
void OpNone::grad(Blob& gradient, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
}

std::vector<size_t> OpNone::computeDim(const std::vector<BlobRef>& args) const {return {};}

void Sum::compute(const vector<BlobRef>& args, Blob& res) const {
    res += args[0].get();
    res += args[1].get();
}
void Sum::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
    res[0].get() += grad;
    res[1].get() += grad;
}

std::vector<size_t> Sum::computeDim(const std::vector<BlobRef>& args) const {
    return {args[0].get().rows, args[0].get().cols};
}

void Multiply::compute(const vector<BlobRef>& args, Blob& res) const {
    for (int i = 0; i < args[0].get().rows; ++i)
        for (int j = 0; j < args[1].get().cols; ++j)
            for (int k = 0; k < args[0].get().cols; ++k)
                res[i][j] += args[0].get()[i][k] * args[1].get()[k][j];
}
void Multiply::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
    for (int i = 0; i < grad.rows; ++i)
        for (int j = 0; j < args[1].get().rows; ++j)
            for (int k = 0; k < grad.cols; ++k)
                res[0].get()[i][j] += grad[i][k] * args[1].get()[j][k];

    for (int i = 0; i < args[0].get().cols; ++i)
        for (int j = 0; j < grad.cols; ++j)
            for (int k = 0; k < grad.rows; ++k)
                res[1].get()[i][j] += args[0].get()[k][i] * grad[k][j];
}

std::vector<size_t> Multiply::computeDim(const std::vector<BlobRef>& args) const {
    return {args[0].get().rows, args[1].get().cols};
}

void ReLU::compute(const vector<BlobRef>& args, Blob& res) const {
    for (int i = 0; i < args[0].get().rows; i++)
        for (int j = 0; j < args[0].get().cols; j++)
            res[i][j] += args[0].get()[i][j] > 0 ? args[0].get()[i][j] : 0;
}
void ReLU::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {

    for (int i = 0; i < args[0].get().rows; i++)
        for (int j = 0; j < args[0].get().cols; j++)
            res[0].get()[i][j] += args[0].get()[i][j] >= 0 ? grad[i][j] : 0;
}

std::vector<size_t> ReLU::computeDim(const std::vector<BlobRef>& args) const {
    return {args[0].get().rows, args[0].get().cols};
}

void BiasSum::compute(const vector<BlobRef>& args, Blob& res) const {
    for (int i = 0; i < args[0].get().rows; i++) {
        for (int j = 0; j < args[0].get().cols; j++) {
            res[i][j] += (args[1].get()[0][j] + args[0].get()[i][j]);
        }
    }
}
void BiasSum::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
    res[0].get() += grad;

    for (int i = 0; i < grad.cols; ++i)
        for (int j = 0; j < grad.rows; ++j)
            res[1].get()[0][i] += grad[j][i];

}

std::vector<size_t> BiasSum::computeDim(const std::vector<BlobRef>& args) const {
    return {args[0].get().rows, args[0].get().cols};
}

void Square::compute(const vector<BlobRef>& args, Blob& res) const {
    for (int i = 0; i < args[0].get().cols; ++i)
        for (int j = 0; j < args[0].get().rows; ++j)
            res[i][j] += args[0].get()[i][j] * args[0].get()[i][j];
}
void Square::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
    for (int i = 0; i < args[0].get().cols; ++i)
        for (int j = 0; j < args[0].get().rows; ++j)
            res[0].get()[i][j] += 2 * args[0].get()[i][j] * grad[i][j];
}

std::vector<size_t> Square::computeDim(const std::vector<BlobRef>& args) const {
    return {args[0].get().rows, args[0].get().cols};
}

void Mean::compute(const vector<BlobRef>& args, Blob& res) const {
    for (int i = 0; i < args[0].get().rows; i++) 
        for (int j = 0; j < args[0].get().cols; j++)
            res[0][0] += args[0].get()[i][j];
    res *= (1.0f / (args[0].get().rows * args[0].get().cols));
}
void Mean::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
    float number = grad[0][0] * (1.0f / (args[0].get().rows * args[0].get().cols));
    for (int i = 0; i < args[0].get().rows; i++) 
        for (int j = 0; j < args[0].get().cols; j++)
            res[0].get()[i][j] += number;
}

std::vector<size_t> Mean::computeDim(const std::vector<BlobRef>& args) const {
    return {1, 1};
}

void Substract::compute(const vector<BlobRef>& args, Blob& res) const {
    for (int i = 0; i < args[0].get().rows; i++) 
        for (int j = 0; j < args[0].get().cols; j++)
            res[i][j] += (args[0].get()[i][j] - args[1].get()[i][j]);
}
void Substract::grad(Blob& grad, const vector<BlobRef>& args, std::vector<BlobRef>& res) const {
    res[0].get() += grad;
    res[1].get() += grad;
    res[1].get() *= -1;
}

std::vector<size_t> Substract::computeDim(const std::vector<BlobRef>& args) const {
    return {args[0].get().rows, args[0].get().cols};
}
