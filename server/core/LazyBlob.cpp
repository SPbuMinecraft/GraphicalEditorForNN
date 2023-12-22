#include <cassert>
#include <vector>
#include <optional>

#include "LazyBlob.h"
#include "Iterations.h"
#include "Allocator.h"
#include "Blob.h"

#define MAX_DIMS_COUNT 4

const Shape& LazyBlob::shape() const {
    if (shape_.has_value()) {
        return *shape_;
    } else {
        initShape();
        return *shape_;
    }
}

LazyBlobView::LazyBlobView(const Blob &ref): ref(ref) {};

void LazyBlobView::initShape() const { 
    shape_ = ref.shape;
};

float LazyBlobView::operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const {
    return ref(k, l, i, j);
}

class LazyBlobUnaryOperation: public LazyBlob {
protected:
    const LazyBlob &a;
    LazyBlobUnaryOperation(const LazyBlob &a): a(a) {};
};

class LazyBlobBinaryOperation: public LazyBlob {
protected:
    const LazyBlob &a, &b;
    LazyBlobBinaryOperation(const LazyBlob &a, const LazyBlob &b): a(a), b(b) {};
};

class LazyBlobStretchableOperation: public LazyBlobBinaryOperation {
protected:
    const BinaryTransform operation;
    LazyBlobStretchableOperation(const LazyBlob &a, const LazyBlob &b,
                                 const BinaryTransform operation):
    LazyBlobBinaryOperation(a, b), operation(operation) {};

    void initShape() const final override { 
        shape_ = Shape {
            {
                a.shape().dim4(),
                a.shape().dim3(), 
                std::max(a.shape().rows(), b.shape().rows()), 
                std::max(a.shape().cols(), b.shape().cols())
            },
            std::max(a.shape().dimsCount, b.shape().dimsCount)
        }; 
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const final override {
        auto ia = i < a.shape().rows() ? i : 0;
        auto ja = j < a.shape().cols() ? j : 0;
        auto ib = i < b.shape().rows() ? i : 0;
        auto jb = j < b.shape().cols() ? j : 0;
        return operation(a(k, l, ia, ja), b(k, l, ib, jb));
    }
};

class LazyBlobTransformOperation: public LazyBlob {
protected:
    const LazyBlob &a;
    LazyBlobTransformOperation(const LazyBlob &a): a(a) {};
};

class LazyBlobReductOperation: public LazyBlobTransformOperation {
protected:
    std::vector<short> axis;
    LazyBlobReductOperation(const LazyBlob &a, std::vector<short> axis): LazyBlobTransformOperation(a), axis(axis) {};

    void initShape() const final override {
        Shape result = a.shape();
        for (auto dim: axis) result[dim] = 1;
        shape_ = result;
    }
};

class LazyBlobSelfSum final: public LazyBlobReductOperation {
public:
    LazyBlobSelfSum(const LazyBlob &a, std::vector<short> axis): LazyBlobReductOperation(a, axis) {};

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        float result = 0;
        
        std::vector<bool> axis = fillAxis(this->axis);
        std::vector<size_t> size = fillSize(axis, a.shape());

        map(size, [&](size_t k1, size_t l1, size_t i1, size_t j1) {
            result += a(
                axis[0] ? k1 : k, 
                axis[1] ? l1 : l, 
                axis[2] ? i1 : i, 
                axis[3] ? j1 : j
            );
        });

        return result;
    }
};

class LazyBlobMean final: public LazyBlobReductOperation {
public:
    LazyBlobMean(const LazyBlob &a, std::vector<short> axis): LazyBlobReductOperation(a, axis) {};

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        float result = 0;
        size_t count = 0;

        std::vector<size_t> axisIndex;
        
        std::vector<bool> axis = fillAxis(this->axis);
        std::vector<size_t> size = fillSize(axis, a.shape());

        map(size, [&](size_t k1, size_t l1, size_t i1, size_t j1) {
            result += a(
                axis[0] ? k1 : k, 
                axis[1] ? l1 : l, 
                axis[2] ? i1 : i, 
                axis[3] ? j1 : j
            );
            count++;
        });

        return result / count;
    }
};

class LazyBlobFill final: public LazyBlobTransformOperation {
public:
    Shape blobShape;
    LazyBlobFill(const LazyBlob &a, Shape shape): LazyBlobTransformOperation(a), blobShape(shape) {};

    void initShape() const final override {
        shape_ = blobShape;
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        Shape realShape = a.shape();
        size_t indices[] = {k, l, i, j};
        size_t realIndices[MAX_DIMS_COUNT];
        for (int i = 0; i < MAX_DIMS_COUNT; i++) {
            if (indices[i] < realShape[i])
                realIndices[i] = indices[i];
            else if (indices[i] >= realShape[i] && realShape[i] == 1 && indices[i] < blobShape[i])
                realIndices[i] = 0;
            else
                assert(false); // if we are trying to fill a dimension that is not equal to one
        }
        return a(realIndices[0], realIndices[1], realIndices[2], realIndices[3]);
    }
};

class LazyBlobSum final: public LazyBlobStretchableOperation {
private:
    static constexpr BinaryTransform plus = [](float x, float y) { return x + y; };
public:
    LazyBlobSum(const LazyBlob &a, const LazyBlob &b): LazyBlobStretchableOperation(a, b, plus) {};
};

class LazyBlobSubtract final: public LazyBlobStretchableOperation {
private:
    static constexpr BinaryTransform minus = [](float x, float y) { return x - y; };
public:
    LazyBlobSubtract(const LazyBlob &a, const LazyBlob &b): LazyBlobStretchableOperation(a, b, minus) {};
};

class LazyBlobMult final: public LazyBlobStretchableOperation {
private:
    static constexpr BinaryTransform multiply = [](float x, float y) { return x * y; };
public:
    LazyBlobMult(const LazyBlob &a, const LazyBlob &b): LazyBlobStretchableOperation(a, b, multiply) {};
};

class LazyBlobDot final: public LazyBlobBinaryOperation {
public:
    LazyBlobDot(const LazyBlob &a, const LazyBlob &b): LazyBlobBinaryOperation(a, b) {};

    void initShape() const final override { 
        shape_ = Shape {
            {a.shape().dim4(), a.shape().dim3(), a.shape().rows(), b.shape().cols()}, 
            std::max(a.shape().dimsCount, b.shape().dimsCount)
        };
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        float result = 0;
        std::vector<size_t> size = {1, 1, 1, a.shape().cols()};
        map(size, [&](size_t k1, size_t l1, size_t i1, size_t j1) {
            result += a(k, l, i, j1) * b(k, l, j1, j);
        });

        return result;
    }
};

class LazyBlobCombine final: public LazyBlobBinaryOperation {
private:
    const BinaryTransform how;
public:
    LazyBlobCombine(const LazyBlob& a, const LazyBlob& b, const BinaryTransform how):
        LazyBlobBinaryOperation(a, b), how(how) {};

    void initShape() const final override { 
        shape_ = a.shape();
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        return how(a(k, l, i, j), b(k, l, i, j));
    }
};

class LazyBlobTranspose final: public LazyBlobUnaryOperation {
public:
    LazyBlobTranspose(const LazyBlob &a): LazyBlobUnaryOperation(a) {};

    void initShape() const final override { 
        shape_ = Shape {{a.shape().dim4(), a.shape().dim3(), a.shape().cols(), a.shape().rows()}, a.shape().dimsCount};
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        return a(k, l, j, i);
    }
};

class LazyBlobApply: public LazyBlobUnaryOperation {
private:
    const UnaryTransform operation;
public:
    LazyBlobApply(const LazyBlob &a, const UnaryTransform operation): LazyBlobUnaryOperation(a), operation(operation) {};

    void initShape() const final override { 
        shape_ = a.shape();
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const final override {
        return operation(a(k, l, i, j));
    }
};

class LazyScalarOperation: public LazyBlobUnaryOperation {
protected:
    float scalar;
    LazyScalarOperation(const LazyBlob &a, float b): LazyBlobUnaryOperation(a), scalar(b) {};

    void initShape() const final override { 
        shape_ = a.shape();
    }
};
class LazyScalarSum: public LazyScalarOperation {
public:
    LazyScalarSum(const LazyBlob &a, float b): LazyScalarOperation(a, b) {};

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        return scalar + a(k, l, i, j);
    }
};
class LazyScalarMult: public LazyScalarOperation {
public:
    LazyScalarMult(const LazyBlob &a, float b): LazyScalarOperation(a, b) {};

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        return scalar * a(k, l, i, j);
    }
};

class LazyBlobNegate final: public LazyBlobApply {
private:
    static constexpr UnaryTransform negateOperation = [](float x) { return -x; };
public:
    LazyBlobNegate(const LazyBlob &a): LazyBlobApply(a, negateOperation) {};
};

bool canStretch(size_t a, size_t b) {
    return a == b || a == 1 || b == 1;
}

void assertStretchable(const LazyBlob &a, const LazyBlob &b) {
    assert(canStretch(a.shape().rows(), b.shape().rows()));
    assert(canStretch(a.shape().cols(), b.shape().cols()));
}

template<class LazyNode>
static inline const LazyBlob& alloc1(const LazyBlob &a) {
    void* location = Allocator::allocateBytes(sizeof(LazyNode));
    return *(new(location) LazyNode(a));
}

template<class LazyNode>
static inline const LazyBlob& alloc2(const LazyBlob &a, const LazyBlob &b) {
    void* location = Allocator::allocateBytes(sizeof(LazyNode));
    return *(new(location) LazyNode(a, b));
}

const LazyBlob& operator + (const LazyBlob &a, const LazyBlob &b) {
    assertStretchable(a, b);
    return alloc2<LazyBlobSum>(a, b);
}

const LazyBlob& operator - (const LazyBlob &a, const LazyBlob &b) {
    assertStretchable(a, b);
    return alloc2<LazyBlobSubtract>(a, b);
}

const LazyBlob& operator * (const LazyBlob &a, const LazyBlob &b) {
    assertStretchable(a, b);
    return alloc2<LazyBlobMult>(a, b);
}

const LazyBlob& LazyBlob::dot(const LazyBlob& a) const {
    assert(shape().cols() == a.shape().rows());
    return alloc2<LazyBlobDot>(*this, a);
}

const LazyBlob& LazyBlob::transposed() const {
    return alloc1<LazyBlobTranspose>(*this);
}

const LazyBlob& LazyBlob::sum(std::vector<short> axis) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobSelfSum));
    return *(new(location) LazyBlobSelfSum(*this, axis));
}

const LazyBlob& LazyBlob::mean(std::vector<short> axis) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobMean));
    return *(new(location) LazyBlobMean(*this, axis));
}

const LazyBlob& LazyBlob::fill(Shape shape) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobFill));
    return *(new(location) LazyBlobFill(*this, shape));
}

const LazyBlob& LazyBlob::applying(const UnaryTransform t) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobApply));
    return *(new(location) LazyBlobApply(*this, t));
}

const LazyBlob& combine(const LazyBlob &a, const LazyBlob &b, const BinaryTransform t) {
    assert(a.shape().rows() == b.shape().rows() && a.shape().cols() == b.shape().cols());
    void* location = Allocator::allocateBytes(sizeof(LazyBlobCombine));
    return *(new(location) LazyBlobCombine(a, b, t));
}

const LazyBlob& operator - (const LazyBlob &a) {
    return alloc1<LazyBlobNegate>(a);
}

const LazyBlob& operator & (const LazyBlob &a, const LazyBlob &b) {
    return a.dot(b);
}

const LazyBlob& operator + (float a, const LazyBlob &b) {
    return b + a;
};
const LazyBlob& operator - (float a, const LazyBlob &b) {
    return b + -a;
};
const LazyBlob& operator * (float a, const LazyBlob &b) {
    return b * a;
};

const LazyBlob& operator + (const LazyBlob &a, float b) {
    void* location = Allocator::allocateBytes(sizeof(LazyScalarSum));
    return *(new(location) LazyScalarSum(a, b));
};
const LazyBlob& operator - (const LazyBlob &a, float b) {
    return a + -b;
};
const LazyBlob& operator * (const LazyBlob &a, float b) {
    void* location = Allocator::allocateBytes(sizeof(LazyScalarMult));
    return *(new(location) LazyScalarMult(a, b));
};
const LazyBlob& operator / (const LazyBlob &a, float b) {
    return a * (1.0 / b);
}

Blob& operator += (Blob& a, const LazyBlob& b) {
    assert(a.shape == b.shape());

    std::vector<size_t> size = a.shape.getDims();
    map(size, [&](size_t k, size_t l, size_t i, size_t j) {
       a(k, l, i, j) += b(k, l, i, j);
    });
    return a;
}

Blob& operator -= (Blob& a, const LazyBlob& b) {
    assert(a.shape == b.shape());

    std::vector<size_t> size = a.shape.getDims();
    map(size, [&](size_t k, size_t l, size_t i, size_t j) {
       a(k, l, i, j) -= b(k, l, i, j);
    });
    return a;
}

Blob& operator *= (Blob& a, const LazyBlob& b) {
    assert(a.shape == b.shape());

    std::vector<size_t> size = a.shape.getDims();
    map(size, [&](size_t k, size_t l, size_t i, size_t j) {
       a(k, l, i, j) *= b(k, l, i, j);
    });
    return a;
}

std::ostream& operator<<(std::ostream& os, const LazyBlob &b) {
    for (int l = 0; l < b.shape().dim4(); ++l) {
        for (int k = 0; k < b.shape().dim3(); ++k){
            for (int i = 0; i < b.shape().rows(); ++i) {
                for (int j = 0; j < b.shape().cols(); ++j)
                    os << b(i, j) << " ";
                os << std::endl;
            }
            os << std::endl;
        }
        os << std::endl;
    }
    return os;
}
