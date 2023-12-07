#pragma once

#include <vector>
#include <string>
#include "UnshuffledDataLoader.h"
#include "Allocator.h"


class UnshuffledCsvLoader: public UnshuffledDataLoader {
private:
    std::vector<std::pair<std::vector<float>, float>> data;
public:
    UnshuffledCsvLoader() = default;
    void load_data(std::string path) override;
    std::pair<Blob, float> operator[](std::size_t index) const override;
    void add_data(std::pair<std::vector<float>, float> instance) override;
    std::size_t size() const override;
    virtual std::pair<std::vector<float>, float> get_raw(std::size_t index) const override;
};
