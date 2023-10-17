#include "Allocator.h"
#include "Blob.h"
#include <iostream>
#include <exception>
#include <cassert>

using namespace std;

Allocator* Allocator::instance = NULL;

void Allocator::start(dict<Shape, size_t> counts, size_t bufSize) {
    assert(!instance);
    instance = new Allocator();

    instance->buf.base = new char[bufSize];
    instance->buf.size = bufSize;

    size_t total = 0;
    for (auto const [shape, count]: counts) {
        total += count * shape.rows * shape.cols;
    }
    float* base = new float[total];

    size_t offset = 0;
    for (auto const [shape, count]: counts) {
        Stack s;
        s.top = 0;
        for (int i = 0; i < count; ++i) {
            float* ptr = base + offset + i * shape.rows * shape.cols;
            s.pointers.push_back(ptr);
            instance->getInfo[ptr] = {shape, i};
        }

        offset += count * shape.rows * shape.cols;
        instance->pointerTable[shape] = s;
    }

    instance->base = base;
}

float* Allocator::allocate(Shape shape) {
    assert(instance);
    Stack &s = instance->pointerTable[shape];
    // ran out of memory
    assert(s.top < s.pointers.size());
    // THIS IS IMPORTANT
    get<1>(instance->getInfo[s.pointers[s.top]]) = s.top;

    float* result = s.pointers[s.top++];
    if (instance->kostyl) {
        instance->sessionPointers.push_back(result);
        instance->kostyl = false;
    }
    return result;
}

void Allocator::release(float* ptr) {
    assert(instance);
//    if (!instance) return; // you are in luck! everything is already released....
    auto const [shape, i] = instance->getInfo[ptr];

    // shape not found
    assert(instance->pointerTable.find(shape) != instance->pointerTable.end());

    Stack &s = instance->pointerTable[shape];
    assert(i < s.top); // pointer is already free

    if (i == --s.top) return;
    swap(s.pointers[i], s.pointers[s.top]);

    // remember for which index we swapped our top pointer
    get<1>(instance->getInfo[s.pointers[i]]) = i;
}

void* Allocator::allocateBytes(size_t size) {
    assert(instance);
    SessionBuffer& buf = instance->buf;
    // ran out of memory
    assert(buf.top + size <= buf.size);
    void* result = instance->buf.base + instance->buf.top;
    buf.top += size;
    return result;
}

Blob* Allocator::allocateBlob(size_t rows, size_t cols) {
    assert(instance);
    void* location = allocateBytes(sizeof(Blob));
    instance->kostyl = true;
    return new(location) Blob(rows, cols);
}

void Allocator::endSession() {
    assert(instance);
    instance->buf.top = 0;
    for (auto ptr: instance->sessionPointers)
        release(ptr);
    instance->sessionPointers.clear();
}

void Allocator::end() {
    assert(instance);
    delete [] instance->buf.base;
    delete [] instance->base;
    delete instance;
    instance = NULL;
}

void Allocator::printStats() { 
    assert(instance);
    printf("Allocator base: 0x%tx\n", (ptrdiff_t)instance->base);
    printf("shapes count: %ld\n", instance->pointerTable.size());
    size_t total = 0, used = 0;
    for (auto const &[shape, s]: instance->pointerTable) {
        total += shape.rows * shape.cols * s.pointers.size();
        used += shape.rows * shape.cols * s.top;
        printf("  (%ld, %ld): %d/%ld pointers occupied\n",
               shape.rows, shape.cols, s.top, s.pointers.size());
    }
    if (instance->buf.top)
        printf("Current session: %ld/%ld bytes used\n", instance->buf.top, instance->buf.size);
    printf("Total %ld/%ld floats used\n", used, total);
}

void Allocator::printPointersInfo() {
    assert(instance);
    printf("Allocator base: 0x%tx\n", (ptrdiff_t)instance->base);
    printf("shapes count: %ld\n", instance->pointerTable.size());
    int used = 0, total = 0;
    for (auto const &[shape, s]: instance->pointerTable) {
        printf("Pointers of shape (%ld, %ld), count = %d:\n", shape.rows, shape.cols, s.top);
        for (int i = 0; i < s.top; ++i) {
            auto const [ignore, j] = instance->getInfo[s.pointers[i]];
            printf("  [%d] = 0x%tx: %d\n", i, (ptrdiff_t)s.pointers[i], j);
        }
        used += s.top;
        total += s.pointers.size();
    }
    printf("Total %d/%d pointers used\n", used, total);
}


bool Shape::operator == (const Shape& other) const {
    return this->rows == other.rows && this->cols == other.cols;
}

inline void hash_combine(std::size_t& seed, size_t v) {
    seed ^= v + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

size_t hash<Shape>::operator()(const Shape& shape) const {
    size_t seed = 1843;

    hash_combine(seed, hash<size_t>()(shape.rows));
    hash_combine(seed, hash<size_t>()(shape.cols));

    return seed;
};
