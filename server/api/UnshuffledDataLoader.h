#pragma once

#include <string>
#include <vector>
#include "Allocator.h"

class UnshuffledDataLoader {
public:
    UnshuffledDataLoader() = default;
    virtual ~UnshuffledDataLoader() = default;
    virtual void load_data(std::string path) = 0;
    virtual std::pair<Blob, float> operator[](std::size_t index) const = 0;
    virtual void add_data(std::pair<std::vector<float>, float> instance) = 0;
    virtual std::size_t size() const = 0;
    virtual std::pair<std::vector<float>, float> get_raw(std::size_t index) const = 0;
};
