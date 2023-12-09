#pragma once

#include <vector>
#include <string>
#include "UnshuffledDataLoader.h"


class UnshuffledCsvLoader: public UnshuffledDataLoader {
private:
    std::vector<std::pair<std::vector<float>, float>> data;
public:
    UnshuffledCsvLoader() = default;
    void load_data(std::string path) override;
    void add_data(const UnshuffledDataLoader* other, int index) override;
    std::size_t size() const override;
    std::pair<std::vector<float>, float> get_raw(std::size_t index) const override;
    Shape get_appropriate_shape(std::size_t index, std::size_t batch_size) const override;
};
