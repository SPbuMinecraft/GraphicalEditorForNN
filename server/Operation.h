#include <vector>
#include "Blob.h"

using namespace std;

struct Operation {
    virtual Blob compute(vector<BlobRef> args) const = 0;
    virtual vector<BlobRef> grad(Blob& gradient, vector<BlobRef> args) const = 0;
    const Operation& operator=(const Operation& other) const;
};

struct Sum: Operation {
    Blob compute(vector<BlobRef> args) const override;
    vector<BlobRef> grad(Blob& gradient, vector<BlobRef> args) const override;
};

struct None: Operation {
    Blob compute(vector<BlobRef> args) const override;
    vector<BlobRef> grad(Blob& gradient, vector<BlobRef> args) const override;
};