#pragma once

#include "UnshuffledDataLoader.h"
#include <vector>

void generate_rearrangement(std::vector<int>& rearrangement, std::size_t size);

class DataLoader {
private:
    UnshuffledDataLoader* loader;
    std::vector<int> rearrangement;
    std::size_t batch_size;
public:
    DataLoader() = default;
    DataLoader(UnshuffledDataLoader* _loader, std::size_t _batch_size);
    DataLoader(UnshuffledDataLoader* _loader, std::size_t _batch_size, std::string path);
    void load_data(std::string path);
    std::pair<Blob, std::vector<float>> operator[](std::size_t index) const;
    void add_data(const DataLoader& other, int index);
    std::size_t size() const;
    std::size_t batch_count() const;
    std::pair<std::vector<float>, std::vector<float>> get_raw(std::size_t index) const;
    void shuffle();
};
