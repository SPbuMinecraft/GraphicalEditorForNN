#pragma once

#include <vector>
#include <string>

#include "Stretch.h"
#include "Allocator.h"
#include "Blob.h"

struct Operation {
    /// Using move semantics here, nothing gets copied
    virtual Blob compute(const std::vector<LazyBlobRef>& args) const = 0;
    /// The only way is to return vector of LazyBlobs here because we want to += to the parent's gradients without any allocations
    virtual std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const = 0;
    virtual Shape computeDim(const std::vector<LazyBlobRef>& args) const = 0;
    const Operation& operator=(const Operation& other) const;
    std::string name = "";

    virtual std::string getName() const = 0;
};

struct Noop: Operation {
    std::string name = "Noop";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct ReLU: Operation {
    std::string name = "ReLU";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Sum: Operation {
    std::string name = "Sum";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Multiply: Operation {
    std::string name = "Multiply";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Divide: Operation {
    std::string name = "Divide";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct BiasSum: Operation {
    mutable std::optional<Stretch*> stretch = std::nullopt;
    std::string name = "BiasSum";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Square: Operation {
    std::string name = "Square";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Root: Operation {
    std::string name = "Root";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Mean: Operation {
    std::vector<short> axis;
    std::string name = "Mean";
    virtual std::string getName() const override;
    bool minusOne = false;
    Mean(std::vector<short> axis);
    Mean(std::vector<short> axis, bool minusOne);
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct SumAxis: Operation {
    std::vector<short> axis;
    std::string name = "SumAxis";
    virtual std::string getName() const override;
    SumAxis(std::vector<short> axis);
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Var: Operation {
    std::vector<short> axis;
    std::string name = "Var";
    virtual std::string getName() const override;
    Var(std::vector<short> axis);
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Substract: Operation {
    std::string name = "Substract";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Conv2D: Operation {
    std::string name = "Conv2D";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Fill: Operation {
    std::string name = "Fill";
    virtual std::string getName() const override;
    mutable std::optional<Shape> shape = std::nullopt;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct EPS: Operation {
    std::string name = "EPS";
    virtual std::string getName() const override;
    const float eps = 1e-5;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Exp: Operation {
    std::string name = "Exp";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct Entropy: Operation {
    std::string name = "Entropy";
    virtual std::string getName() const override;
    int classCount;
    Entropy(int classCouont): classCount(classCouont) {};
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};

struct MaxPoolOp: Operation {
    std::string name = "MaxPool";
    virtual std::string getName() const override;
    Blob compute(const std::vector<LazyBlobRef>& args) const override;
    std::vector<LazyBlobRef> grad(const Blob& gradient, const std::vector<LazyBlobRef>& args) const override;
    Shape computeDim(const std::vector<LazyBlobRef>& args) const override;
};
