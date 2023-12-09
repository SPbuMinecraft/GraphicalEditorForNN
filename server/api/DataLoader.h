#pragma once

#include "UnshuffledDataLoader.h"
#include <vector>

void generate_rearrangement(std::vector<int>& rearrangement, std::size_t size);

class DataLoader {
private:
    UnshuffledDataLoader* loader;
    std::vector<int> rearrangement;
public:
    DataLoader() = default;
    DataLoader(UnshuffledDataLoader* _loader);
    DataLoader(UnshuffledDataLoader* _loader, std::string path);
    void load_data(std::string path);
    std::pair<Blob, float> operator[](std::size_t index) const;
    void add_data(const DataLoader& other, int index);
    std::size_t size() const;
    std::pair<std::vector<float>, float> get_raw(std::size_t index) const;
};
