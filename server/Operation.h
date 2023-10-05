#pragma once

#include <vector>
#include "Blob.h"

struct Operation {
    virtual Blob compute(std::vector<BlobRef> args) const = 0;
    virtual std::vector<BlobRef> grad(Blob& gradient, std::vector<BlobRef> args) const = 0;
    const Operation& operator=(const Operation& other) const;
};

struct Sum: Operation {
    Blob compute(std::vector<BlobRef> args) const override;
    std::vector<BlobRef> grad(Blob& gradient, std::vector<BlobRef> args) const override;
};

struct None: Operation {
    Blob compute(std::vector<BlobRef> args) const override;
    std::vector<BlobRef> grad(Blob& gradient, std::vector<BlobRef> args) const override;
};