#pragma once

#include "Blob.h"
#include <vector>

struct Operation {
    virtual Blob compute(const std::vector<BlobRef>& args) const = 0;
    virtual std::vector<Blob> grad(Blob& gradient, const std::vector<BlobRef>& args) const = 0;
    const Operation& operator=(const Operation& other) const;
};

struct None: Operation {
    Blob compute(const std::vector<BlobRef>& args) const override;
    std::vector<Blob> grad(Blob& gradient, const std::vector<BlobRef>& args) const override;
};

struct Sum: Operation {
    Blob compute(const std::vector<BlobRef>& args) const override;
    std::vector<Blob> grad(Blob& gradient, const std::vector<BlobRef>& args) const override;
};

struct Multiply: Operation {
    Blob compute(const std::vector<BlobRef>& args) const override;
    std::vector<Blob> grad(Blob& gradient, const std::vector<BlobRef>& args) const override;
};

struct Loss: Operation {
    Blob compute(const std::vector<BlobRef>& args) const override;
    std::vector<Blob> grad(Blob& gradient, const std::vector<BlobRef>& args) const override;
};
