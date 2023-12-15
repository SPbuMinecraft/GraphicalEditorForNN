#include "Allocator.h"
#include "Blob.h"
#include <iostream>
#include <exception>
#include <cassert>
#include <cstring>

using namespace std;

Allocator* Allocator::instance = NULL;

#define TMP_BUFF_SIZE 16384

void Allocator::startVirtualMode() {
    assert(!instance);
    instance = new Allocator();
    instance->buf.base = new char[TMP_BUFF_SIZE];
    instance->buf.size = TMP_BUFF_SIZE;
}

void Allocator::endVirtualMode() {
    assert(instance);
    assert(instance->virtualMode);
    instance->virtualMode = false;

    for (auto &[ptr, stack] : instance->info.pointerTable) {
        for (auto ptr : stack.pointers) {
            delete[] ptr;
        }
        stack.pointers.clear();
    }
    Allocator::endSession();

    instance->info = instance->realInfo;
    start(instance->counts);
}


void Allocator::start(dict<Shape, size_t> counts) {
    size_t total = 0;
    for (auto const &[shape, count]: counts) {
        total += count * shape.size();
    }
    float* base = new float[total];

    size_t offset = 0;
    for (auto const &[shape, count]: counts) {
        Stack s;
        for (int i = 0; i < count; ++i) {
            float* ptr = base + offset + i * shape.size();
            s.pointers.push_back(ptr);
            instance->info.getInfo[ptr] = {shape, i};
        }

        offset += count * shape.size();
        instance->info.pointerTable[shape] = s;
    }

    instance->base = base;
}

float* Allocator::allocate(Shape shape, bool constMemory) {
    assert(instance);
    auto &info = instance->info;
    if (constMemory) {
        float* result = new float[shape.size()];
        instance->constMems.insert(result);
        return result;
    }
    if (instance->virtualMode) {
        instance->counts[shape]++;
        Stack &s = info.pointerTable[shape];
        float* ptr = new float[shape.size()];
        info.getInfo[ptr] = {shape, s.pointers.size()};
        s.pointers.push_back(ptr);
    }
    Stack &s = info.pointerTable[shape];
    // ran out of memory
    assert(s.top < s.pointers.size());
    // THIS IS IMPORTANT
    get<1>(info.getInfo[s.pointers[s.top]]) = s.top;

    float* result = s.pointers[s.top++];
    if (instance->kostyl) {
        instance->sessionPointers.push_back(result);
        instance->kostyl = false;
    }
    return result;
}

void Allocator::release(float* ptr) {
    assert(instance);
    auto &info = instance->info;
    if (instance->constMems.find(ptr) != instance->constMems.end()) {
        delete[] ptr;
        instance->constMems.erase(ptr);
        return;
    }
//    if (!instance) return; // you are in luck! everything is already released....
    auto const [shape, i] = info.getInfo[ptr];

    // shape not found
    assert(info.pointerTable.find(shape) != info.pointerTable.end());

    Stack &s = info.pointerTable[shape];
    assert(i < s.top); // pointer is already free

    if (i == --s.top) return;
    swap(s.pointers[i], s.pointers[s.top]);

    // remember for which index we swapped our top pointer
    get<1>(info.getInfo[s.pointers[i]]) = i;
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

Blob* Allocator::allocateBlob(Shape shape) {
    assert(instance);
    void* location = allocateBytes(sizeof(Blob));
    instance->kostyl = true;
    return new(location) Blob(shape);
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
    if (instance->virtualMode)
        Allocator::endVirtualMode();
    delete [] instance->buf.base;
    delete [] instance->base;
    for (auto &memory : instance->constMems) {
        delete[] memory;
    }
    delete instance;
    instance = NULL;
}

void Allocator::printStats() { 
    assert(instance);
    printf("Allocator base: 0x%tx\n", (ptrdiff_t)instance->base);
    printf("Virtual Mode: %d\n", instance->virtualMode);
    printf("shapes count: %ld\n", instance->info.pointerTable.size());
    size_t total = 0, used = 0;
    for (auto const &[shape, s]: instance->info.pointerTable) {
        total += shape.size() * s.pointers.size();
        used += shape.size() * s.top;
        printf("  (%s): %d/%ld pointers occupied\n",
               shape.toString().c_str(), s.top, s.pointers.size());
    }
    if (instance->buf.top)
        printf("Current session: %ld/%ld bytes used\n", instance->buf.top, instance->buf.size);
    printf("Total %ld/%ld floats used\n", used, total);
}

void Allocator::printPointersInfo() {
    assert(instance);
    printf("Allocator base: 0x%tx\n", (ptrdiff_t)instance->base);
    printf("Virtual Mode: %d\n", instance->virtualMode);
    printf("shapes count: %ld\n", instance->info.pointerTable.size());
    int used = 0, total = 0;
    for (auto const &[shape, s]: instance->info.pointerTable) {
        printf("Pointers of shape (%s), count = %d:\n", shape.toString().c_str(), s.top);
        for (int i = 0; i < s.top; ++i) {
            auto const [ignore, j] = instance->info.getInfo[s.pointers[i]];
            printf("  [%d] = 0x%tx: %d\n", i, (ptrdiff_t)s.pointers[i], j);
        }
        used += s.top;
        total += s.pointers.size();
    }
    printf("Total %d/%d pointers used\n", used, total);
}
