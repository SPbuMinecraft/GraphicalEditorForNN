#pragma once

#include <vector>
#include <string>

#include "Blob.h"

struct Operation {
    virtual void compute(const std::vector<BlobRef>& args, Blob& res) const = 0;
    virtual void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const = 0;
    virtual std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const = 0;
    const Operation& operator=(const Operation& other) const;
};

struct OpNone: Operation {
    std::string name = "OpNone";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct Sum: Operation {
    std::string name = "Sum";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct Multiply: Operation {
    std::string name = "Multiply";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct ReLU: Operation {
    std::string name = "ReLU";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct BiasSum: Operation {
    std::string name = "BiasSum";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct Square: Operation {
    std::string name = "Square";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct Mean: Operation {
    std::string name = "Mean";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};

struct Substract: Operation {
    std::string name = "Substract";
    void compute(const std::vector<BlobRef>& args, Blob& res) const override;
    void grad(Blob& gradient, const std::vector<BlobRef>& args, std::vector<BlobRef>& res) const override;
    std::vector<size_t> computeDim(const std::vector<BlobRef>& args) const override;
};