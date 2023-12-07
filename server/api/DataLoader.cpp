#include <algorithm>
#include <random>
#include <stdexcept>
#include "DataLoader.h"
#include "Blob.h"

void generate_rearrangement(std::vector<int>& rearrangement, std::size_t size) {
    rearrangement.resize(size);
    for (int i = 0; i < rearrangement.size(); ++i) {
        rearrangement[i] = i;
    }
    // Some shuffle magic from StackOverflow
    auto rd = std::random_device {};
    auto rng = std::default_random_engine { rd() };
    std::shuffle(rearrangement.begin(), rearrangement.end(), rng);
}

DataLoader::DataLoader(UnshuffledDataLoader* _loader): loader(_loader) {
    generate_rearrangement(rearrangement, loader->size());
}

DataLoader::DataLoader(UnshuffledDataLoader* _loader, std::string path): loader(_loader) {
    loader->load_data(path);
    generate_rearrangement(rearrangement, loader->size());
}

void DataLoader::load_data(std::string path) {
    loader->load_data(path);
}

std::pair<Blob, float> DataLoader::operator[](std::size_t index) const {
    if (index >= loader->size()) {
        throw std::out_of_range("Index out of range");
    }
    return (*loader)[rearrangement[index]];
}

std::size_t DataLoader::size() const {
    return loader->size();
}

void DataLoader::add_data(std::pair<std::vector<float>, float> instance) {
    loader->add_data(instance);
}

std::pair<std::vector<float>, float> DataLoader::get_raw(std::size_t index) const {
    if (index >= loader->size()) {
        throw std::out_of_range("Index out of range");
    }
    return loader->get_raw(index);
}
