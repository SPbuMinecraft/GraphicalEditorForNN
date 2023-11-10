#include "LazyBlob.h"
#include "Allocator.h"
#include "Blob.h"
#include <cassert>

LazyBlobView::LazyBlobView(const Blob &ref): ref(ref) {};

std::size_t LazyBlobView::rows() const { return ref.rows; };
std::size_t LazyBlobView::cols() const { return ref.cols; };

float LazyBlobView::operator() (std::size_t i, std::size_t j) const {
    return ref(i, j);
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

    std::size_t rows() const final override { return std::max(a.rows(), b.rows()); }
    std::size_t cols() const final override { return std::max(a.cols(), b.cols()); }

    float operator() (std::size_t i, std::size_t j) const final override {
        auto ia = i < a.rows() ? i : 0;
        auto ja = j < a.cols() ? j : 0;
        auto ib = i < b.rows() ? i : 0;
        auto jb = j < b.cols() ? j : 0;
        return operation(a(ia, ja), b(ib, jb));
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

    std::size_t rows() const override { return a.rows(); }
    std::size_t cols() const override { return b.cols(); }

    float operator() (std::size_t i, std::size_t j) const override {
        float result = 0;
        for (int k = 0; k < a.cols(); ++k)
            result += a(i, k) * b(k, j);
        return result;
    }
};

class LazyBlobCombine final: public LazyBlobBinaryOperation {
private:
    const BinaryTransform how;
public:
    LazyBlobCombine(const LazyBlob& a, const LazyBlob& b, const BinaryTransform how):
        LazyBlobBinaryOperation(a, b), how(how) {};

    std::size_t rows() const override { return a.rows(); }
    std::size_t cols() const override { return a.cols(); }

    float operator() (std::size_t i, std::size_t j) const override {
        return how(a(i, j), b(i, j));
    }
};

class LazyBlobTranspose final: public LazyBlobUnaryOperation {
public:
    LazyBlobTranspose(const LazyBlob &a): LazyBlobUnaryOperation(a) {};

    std::size_t rows() const override { return a.cols(); }
    std::size_t cols() const override { return a.rows(); }

    float operator() (std::size_t i, std::size_t j) const override {
        return a(j, i);
    }
};

class LazyBlobApply: public LazyBlobUnaryOperation {
private:
    const UnaryTransform operation;
public:
    LazyBlobApply(const LazyBlob &a, const UnaryTransform operation): LazyBlobUnaryOperation(a), operation(operation) {};

    std::size_t rows() const final override { return a.rows(); }
    std::size_t cols() const final override { return a.cols(); }

    float operator() (std::size_t i, std::size_t j) const final override {
        return operation(a(i, j));
    }
};

class LazyScalarOperation: public LazyBlobUnaryOperation {
protected:
    float scalar;
    LazyScalarOperation(const LazyBlob &a, float b): LazyBlobUnaryOperation(a), scalar(b) {};
};
class LazyScalarSum: public LazyScalarOperation {
public:
    LazyScalarSum(const LazyBlob &a, float b): LazyScalarOperation(a, b) {};

    std::size_t rows() const override { return a.rows(); }
    std::size_t cols() const override { return a.cols(); }

    float operator() (std::size_t i, std::size_t j) const override {
        return scalar + a(i, j);
    }
};
class LazyScalarMult: public LazyScalarOperation {
public:
    LazyScalarMult(const LazyBlob &a, float b): LazyScalarOperation(a, b) {};

    std::size_t rows() const override { return a.rows(); }
    std::size_t cols() const override { return a.cols(); }

    float operator() (std::size_t i, std::size_t j) const override {
        return scalar * a(i, j);
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
    assert(canStretch(a.rows(), b.rows()));
    assert(canStretch(a.cols(), b.cols()));
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
    assert(cols() == a.rows());
    return alloc2<LazyBlobDot>(*this, a);
}

const LazyBlob& LazyBlob::transposed() const {
    return alloc1<LazyBlobTranspose>(*this);
}

const LazyBlob& LazyBlob::applying(const UnaryTransform t) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobApply));
    return *(new(location) LazyBlobApply(*this, t));
}

const LazyBlob& combine(const LazyBlob &a, const LazyBlob &b, const BinaryTransform t) {
    assert(a.rows() == b.rows() && a.cols() == b.cols());
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
    // either equal or can stretch b
    assert(a.rows == b.rows() || b.rows() == 1);
    assert(a.cols == b.cols() || b.cols() == 1);
    size_t rows = b.rows(), cols = b.cols();
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            a[i][j] += b(i < rows ? i : 0, j < cols ? j : 0);
    return a;
}
Blob& operator -= (Blob& a, const LazyBlob& b) {
    assert(a.rows == b.rows() || b.rows() == 1);
    assert(a.cols == b.cols() || b.cols() == 1);
    size_t rows = b.rows(), cols = b.cols();
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            a[i][j] -= b(i < rows ? i : 0, j < cols ? j : 0);
    return a;
}
Blob& operator *= (Blob& a, const LazyBlob& b) {
    assert(a.rows == b.rows() || b.rows() == 1);
    assert(a.cols == b.cols() || b.cols() == 1);
    size_t rows = b.rows(), cols = b.cols();
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            a[i][j] *= b(i < rows ? i : 0, j < cols ? j : 0);
    return a;
}

std::ostream& operator<<(std::ostream& os, const LazyBlob &b) {
    for (int i = 0; i < b.rows(); ++i) {
        for (int j = 0; j < b.cols(); ++j)
            os << b(i, j) << " ";
        os << std::endl;
    }
    return os;
}
