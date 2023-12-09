#include <stdexcept>
#include "UnshuffledCsvLoader.h"
#include "CsvLoader.h"
#include "Allocator.h"
#include "Blob.h"

void UnshuffledCsvLoader::load_data(std::string path) {
    data.clear();
    auto file_data = CsvLoader::load_csv(path);
    data.resize(file_data.size());
    for (int i = 0; i < file_data.size(); ++i) {
        float result = file_data[i].back();
        file_data[i].pop_back();
        data[i] = {file_data[i], result};
    }
}

std::pair<Blob, float> UnshuffledCsvLoader::operator[](std::size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return {Blob::constBlob(Shape({0, 0, 1, data[index].first.size()}), data[index].first.data()), data[index].second};
}

void UnshuffledCsvLoader::add_data(const UnshuffledDataLoader* other, int index) {
    data.push_back(other->get_raw(index));
}

std::size_t UnshuffledCsvLoader::size() const {
    return data.size();
}

std::pair<std::vector<float>, float> UnshuffledCsvLoader::get_raw(std::size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return {data[index].first, data[index].second};
}