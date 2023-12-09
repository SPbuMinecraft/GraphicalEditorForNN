#pragma once

#include "UnshuffledDataLoader.h"
#include "Blob.h"
#include <vector>
#include <string>

class UnshuffledImgLoader: public UnshuffledDataLoader {
private:
    std::vector<std::pair<std::string, float>> data;
public:
    UnshuffledImgLoader() = default;
    void load_data(std::string path) override; // path to folder
    std::pair<Blob, float> operator[](std::size_t index) const override;
    void add_data(const UnshuffledDataLoader* other, int index) override;
    std::size_t size() const override;
    virtual std::pair<std::vector<float>, float> get_raw(std::size_t index) const override;
};
