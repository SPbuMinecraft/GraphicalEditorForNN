#include <algorithm>
#include <random>
#include <stdexcept>
#include "DataLoader.h"
#include "Blob.h"
#include "Allocator.h"

void generate_rearrangement(std::vector<int>& rearrangement, std::size_t size) {
    rearrangement.resize(size);
    for (int i = 0; i < rearrangement.size(); ++i) {
        rearrangement[i] = i;
    }
    // Some shuffle magic from StackOverflow
    auto rng = std::default_random_engine { 32 };
    std::shuffle(rearrangement.begin(), rearrangement.end(), rng);
}

DataLoader::DataLoader(UnshuffledDataLoader* _loader, std::size_t _batch_size): loader(_loader), batch_size(_batch_size) {
    generate_rearrangement(rearrangement, loader->size());
}

DataLoader::DataLoader(UnshuffledDataLoader* _loader, std::size_t _batch_size, std::string path): loader(_loader), batch_size(_batch_size) {
    loader->load_data(path);
    generate_rearrangement(rearrangement, loader->size());
}

void DataLoader::load_data(std::string path) {
    loader->load_data(path);
}

std::pair<Blob, std::vector<float>> DataLoader::operator[](std::size_t index) const { // batch_size lines from index
    if (index >= loader->size()) {
        throw std::out_of_range("Index out of range");
    }
    auto data = get_raw(index);
    Shape shape = loader->get_appropriate_shape(index, batch_size);
    return {Blob::constBlob(shape, data.first.data()), data.second};
}

std::size_t DataLoader::size() const {
    return (loader->size() + batch_size - 1) / batch_size;
}

void DataLoader::add_data(const DataLoader& other, int index) {
    loader->add_data(other.loader, index);
}

std::pair<std::vector<float>, std::vector<float>> DataLoader::get_raw(std::size_t batch_index) const { // batch_size lines from index
    if (batch_index >= size()) {
        throw std::out_of_range("Index out of range");
    }
    std::vector<float> data;
    std::vector<float> res(batch_size, 0);
    Shape shape = loader->get_appropriate_shape(rearrangement[batch_index], batch_size);
    auto dims = shape.getDims();
    data.resize(shape.size(), 0);
    int cur_data = 0;
    for (int i = batch_size * batch_index; i < (batch_size + 1) * batch_index; ++i) {
        if (i >= loader->size()) {
            break;
        }
        auto line = loader->get_raw(rearrangement[i]);
        res[i - batch_size * batch_index] = line.second;
        for (int j = 0; j < line.first.size(); ++j) {
            data[cur_data++] = line.first[j];
        }
    }
    return {std::move(data), std::move(res)};
}

void DataLoader::shuffle() {
    generate_rearrangement(rearrangement, loader->size());
}
