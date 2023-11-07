#include <cassert>
#include <vector>

#include "Shape.h"
#include "Iterations.h"
#include "LazyBlob.h"
#include "Allocator.h"
#include "Blob.h"

LazyBlobView::LazyBlobView(const Blob &ref): ref(ref) {};

Shape LazyBlobView::shape() const { return ref.shape; };

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

    Shape shape() const final override { 
        return Shape {
            a.shape().dim4,
            a.shape().dim3, 
            std::max(a.rows(), b.rows()), 
            std::max(a.cols(), b.cols()), 
            std::max(a.shape().dimsCount, b.shape().dimsCount)
        }; 
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const final override {
        auto ia = i < a.rows() ? i : 0;
        auto ja = j < a.cols() ? j : 0;
        auto ib = i < b.rows() ? i : 0;
        auto jb = j < b.cols() ? j : 0;
        return operation(a(k, l, ia, ja), b(k, l, ib, jb));
    }
};

class LazyBlobSelfOperation: public LazyBlob {
protected:
    const LazyBlob &a;
    LazyBlobSelfOperation(const LazyBlob &a): a(a) {};
};

class LazyBlobSelfSum final: public LazyBlobSelfOperation {
public:
    std::vector<size_t> axis;
    LazyBlobSelfSum(const LazyBlob &a, std::vector<size_t> axis): LazyBlobSelfOperation(a), axis(axis) {};

    Shape shape() const final override {
        std::size_t realShape[4];
        memcpy(realShape, a.shape().dims, 4 * sizeof(std::size_t));
        for (auto dim: axis)
            realShape[dim] = 1;
        return Shape {realShape[0], realShape[1], realShape[2], realShape[3], a.shape().dimsCount};
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        float result = 0;
        // std::vector<size_t> axisIndex;
        size_t indexes[] = {k, l, i, j};
        // for (int i = 0; i < 4 - shape().dimsCount; ++i) {
        //     axisIndex.push_back(0);
        // }
        // for (int i = 0; axis.size() < 4; i++) {
        //     axisIndex.push_back(indexes[i]);
        // }
        
        bool axis[] = {false, false, false, false};
        size_t size[] = {0, 0, 0, 0};
        for (auto dim: this->axis) {
            axis[dim] = true;
            size[dim] = a.shape().dims[dim];
        }
        opp(axis, size, [&](size_t k1, size_t l1, size_t i1, size_t j1) {
            result += a(axis[0] ? k1 : indexes[0], axis[1] ? l1 : indexes[1], axis[2] ? i1 : indexes[2], axis[3] ? l1 : indexes[3]);
            // result += a(axis[0] ? k1 : axisIndex[0], axis[1] ? l1 : axisIndex[1], axis[2] ? i1 : axisIndex[2], axis[3] ? l1 : axisIndex[3]);
        });

        return result;
    }
};

class LazyBlobSelfMean final: public LazyBlobSelfOperation {
public:
    std::vector<size_t> axis;
    LazyBlobSelfMean(const LazyBlob &a, std::vector<size_t> axis): LazyBlobSelfOperation(a), axis(axis) {};

    Shape shape() const final override {
        std::size_t realShape[4];
        memcpy(realShape, a.shape().dims, 4 * sizeof(std::size_t));
        for (auto dim: axis)
            realShape[dim] = 1;
        return Shape {realShape[0], realShape[1], realShape[2], realShape[3], a.shape().dimsCount};
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        float result = 0;
        size_t count = 0;

        std::vector<size_t> axisIndex;
        size_t indexes[] = {k, l, i, j};
        // for (int i = 0; i < 4 - shape().dimsCount; ++i) {
        //     axisIndex.push_back(0);
        // }
        // for (int i = 0; axisIndex.size() < 4; i++) {
        //     axisIndex.push_back(indexes[i]);
        // }
        
        bool axis[] = {false, false, false, false};
        size_t size[] = {0, 0, 0, 0};
        for (auto dim: this->axis) {
            axis[dim] = true;
            size[dim] = a.shape().dims[dim];
        }
        opp(axis, size, [&](size_t k1, size_t l1, size_t i1, size_t j1) {
            result += a(axis[0] ? k1 : indexes[0], axis[1] ? l1 : indexes[1], axis[2] ? i1 : indexes[2], axis[3] ? l1 : indexes[3]);
            // result += a(axis[0] ? k1 : axisIndex[0], axis[1] ? l1 : axisIndex[1], axis[2] ? i1 : axisIndex[2], axis[3] ? l1 : axisIndex[3]);
            count++;
        });

        return result / count;
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

    Shape shape() const final override { 
        return Shape {a.shape().dim4, a.shape().dim3, a.rows(), b.cols(), std::max(a.shape().dimsCount, b.shape().dimsCount)};
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        float result = 0;

        bool axis[] = {false, false, false, true};
        size_t size[] = {0, 0, 0, a.cols()};
        opp(axis, size, [&](size_t i1, size_t j1, size_t k1, size_t l1) {
            result += a(k, l, i, l1) * b(k, l, l1, j);
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

    Shape shape() const final override { 
        return a.shape();
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        return how(a(k, l, i, j), b(k, l, i, j));
    }
};

class LazyBlobTranspose final: public LazyBlobUnaryOperation {
public:
    LazyBlobTranspose(const LazyBlob &a): LazyBlobUnaryOperation(a) {};

    Shape shape() const final override { 
        return Shape {a.shape().dim4, a.shape().dim3, a.cols(), a.rows(), a.shape().dimsCount};
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

    Shape shape() const final override { 
        return a.shape();
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const final override {
        return operation(a(k, l, i, j));
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

    Shape shape() const final override { 
        return a.shape();
    }

    float operator() (std::size_t k, std::size_t l, std::size_t i, std::size_t j) const override {
        return scalar + a(k, l, i, j);
    }
};
class LazyScalarMult: public LazyScalarOperation {
public:
    LazyScalarMult(const LazyBlob &a, float b): LazyScalarOperation(a, b) {};

    Shape shape() const final override { 
        return a.shape();
    }

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

const LazyBlob& LazyBlob::sum(std::vector<std::size_t> axis) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobSelfSum));
    return *(new(location) LazyBlobSelfSum(*this, axis));
}

const LazyBlob& LazyBlob::mean(std::vector<std::size_t> axis) const {
    void* location = Allocator::allocateBytes(sizeof(LazyBlobSelfMean));
    return *(new(location) LazyBlobSelfMean(*this, axis));
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
    assert(a.shape.rows == b.rows() || b.rows() == 1);
    assert(a.shape.cols == b.cols() || b.cols() == 1);
    size_t rows = b.rows(), cols = b.cols();
    
    bool axis[] = {false, false, true, true};
    size_t size[] = {0, 0, a.shape.rows, a.shape.cols};
    opp(axis, size, [&](size_t i, size_t j, size_t k, size_t l) {
        *(a.get_address(i, j)) += b(k < rows ? k : 0, l < cols ? l : 0);
    });

    return a;
}
Blob& operator -= (Blob& a, const LazyBlob& b) {
    assert(a.shape.rows == b.rows() || b.rows() == 1);
    assert(a.shape.cols == b.cols() || b.cols() == 1);
    size_t rows = b.rows(), cols = b.cols();
    
    bool axis[] = {false, false, true, true};
    size_t size[] = {0, 0, a.shape.rows, a.shape.cols};
    opp(axis, size, [&](size_t i, size_t j, size_t k, size_t l) {
        *(a.get_address(k, l)) -= b(k < rows ? k : 0, l < cols ? l : 0);
    });

    return a;
}
Blob& operator *= (Blob& a, const LazyBlob& b) {
    assert(a.shape.rows == b.rows() || b.rows() == 1);
    assert(a.shape.cols == b.cols() || b.cols() == 1);
    size_t rows = b.rows(), cols = b.cols();
    
    bool axis[] = {false, false, true, true};
    size_t size[] = {0, 0, a.shape.rows, a.shape.cols};
    opp(axis, size, [&](size_t i, size_t j, size_t k, size_t l) {
        *(a.get_address(k, l)) *= b(k < rows ? k : 0, l < cols ? l : 0);
    });
    return a;
}

std::ostream& operator<<(std::ostream& os, const LazyBlob &b) {
    bool axis[] = {false, false, true, true};
    size_t size[] = {0, 0, b.rows(), b.cols()};
    opp(axis, size, [&](size_t i, size_t j, size_t k, size_t l) {
        os << b(k, l) << " ";
    });
    return os;
}
