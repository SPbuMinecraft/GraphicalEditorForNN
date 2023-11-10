#pragma once

#include <vector>
#include <string>

#include "Allocator.h"
#include "Blob.h"

struct Operation {
    /// Using move semantics here, nothing gets copied
    virtual Blob compute(const std::vector<LazyBlobRef>& args) const = 0;
    /// The only way is to return vector of LazyBlobs here because we want to += to the parent's gradients without any allocations
    virtual std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const = 0;
    virtual Shape computeDim(const std::vector<LazyBlobRef>& args) const = 0;
    const Operation& operator=(const Operation& other) const;
};

struct Noop: Operation {
    std::string name = "Noop";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct ReLU: Operation {
    std::string name = "ReLU";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Sum: Operation {
    std::string name = "Sum";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Multiply: Operation {
    std::string name = "Multiply";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct BiasSum: Operation {
    std::string name = "BiasSum";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Square: Operation {
    std::string name = "Square";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Mean: Operation {
    std::string name = "Mean";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Substract: Operation {
    std::string name = "Substract";
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};
