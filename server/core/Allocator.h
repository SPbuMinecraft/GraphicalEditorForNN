#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "Shape.h"

class Blob;
class Allocator;

class Stack {
    int top;
    std::vector<float*> pointers;

    friend Allocator;
public:
    Stack(): top(0) {};
};

template<>
struct std::hash<Shape> {
    std::size_t operator()(const Shape& k) const;
};

class SessionBuffer {
    char* base = NULL;
    std::size_t top = 0;
    std::size_t size = 0;

    friend Allocator;
};

typedef std::tuple<Shape, int> PtrInfo;

template<typename Key, typename Value>
using dict = std::unordered_map<Key, Value>;

struct AllocationInfo {
    dict<Shape, Stack> pointerTable;
    dict<float*, PtrInfo> getInfo;
};

/** Singleton class
 - Description:
 Allocated enough storage, then gives on demand fast.

- Usage:
 ```
 dict<Shape, size_t> counts = { {{3, 4}, 10}, {{5, 8}, 1} };
 Allocator::start(counts);

 float* data = Allocator::allocate({5, 8}); // ok
 float* data2 = Allocator::allocate({3, 4}); // ok
 // float* data3 = Allocator::allocate({5, 8}); // error, not enough space
 Allocator::release(data); // ok
 float* data3 = Allocator::allocate({5, 8}); // ok

 Allocator::end();
 ```

 Additionaly supports session mode:
 - You strart a session by calling `allocateBytes`
 - During session you can quickly allocate memory with more `allocateBytes` calls
 - End session by calling `endSession`(wow)
 - When session's ended, all space allocated in that session DEALLOCATES, you CANNOT use it anymore
 */
class Allocator {
    float* base = NULL;

    bool kostyl = false;

    static Allocator *instance;

    std::vector<float*> sessionPointers;
    SessionBuffer buf;
    std::unordered_set<float*> constMems;

    bool virtualMode = true;
    dict<Shape, std::size_t> counts;

    AllocationInfo realInfo;
    AllocationInfo virtInfo;
    AllocationInfo& info = virtInfo;

    Allocator() {};
public:

    static void startVirtualMode();
    static void endVirtualMode();
    /// Call this **BEFORE** calling `allocate`
    /// For now, just need to pre-count all shapes, maybe later there will be a virtual mode,
    /// where all shapes will be counted automatically after a dry run through the graph.
    /// - Parameter counts: **UPPER** limit to the needed amount of EACH shape
    /// - Parameter sessionBufferSize: max bytesize of a buffer you can alloc and free multiple times
    static void start(dict<Shape, std::size_t> counts);

    /// Only call this after `start` and **before** `end`
    /// Gives a pointer to a memory big enough to store the requested shape
    /// - Parameter shape: The requested shape itself
    static float* allocate(Shape shape, bool constMemory = false);

    /// Call this when your object is done using the pointer
    /// - Parameter ptr: A pointer that you **GOT** from the `allocate` function
    /// - Warning: **DO NOT** pass already freed pointer, or call this func after calling `end` and before calling `start`
    /// - Warning: Only call with a pointer you got from the `allocate` function!
    static void release(float* ptr);

    /// Call this to start a session or during a session
    /// Allocates a pointer big enough to store `size` bytes
    /// Usage: `Obj* arr = (Obj*)allocateBytes(count * sizeof(Obj));`
    /// - Parameter size: a size in **BYTES**
    /// - Warning: Only call after `start` and before `end`
    static void* allocateBytes(std::size_t size);

    /// It will allocate you a space for the actual Blob + the blobs data of the required shape
    /// All of this will be deallocated when the next `endSession()` gets called
    /// Needs to be removed, for now stays because I don't know how to solve this fckng problem
    /// - Parameter rows: amount of rows for the new blob
    /// - Parameter cols: amount of cols for the new blob
    /// - Warning: `(rows, cols)` shape must be registered in the `start` method
    /// - Returns: Pointer to the new blob
    static Blob* allocateBlob(Shape shape);

    /// Call this to end a session, if session haven't started it's a noop
    /// - Warning: Only call after `start` and before `end`
    static void endSession();

    /// Utility function, prints current level of memory usage
    /// - Warning: Only call after `start` and before `end`
    static void printStats();

    /// Utility, prints debug info about allocated pointers and their `refCounts`
    /// - Warning: Only call after `start` and before `end`
    static void printPointersInfo();

    /// Call this when you are done, **DO NOT FORGET**
    /// **ONLY** after calling this, you can call `start` again
    static void end();
};
