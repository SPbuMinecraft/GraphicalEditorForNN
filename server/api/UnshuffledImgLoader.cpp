#include "UnshuffledImgLoader.h"
#include "CsvLoader.h"
#include "Blob.h"
#include "Allocator.h"
#include "ImageLoader.h"
#include <filesystem>
#include <string>

void UnshuffledImgLoader::load_data(std::string path) {
    for (auto const& dir_entry : std::filesystem::recursive_directory_iterator(path)) {
        std::string file_path = dir_entry.path();
        if (file_path.size() >= 4 && file_path.substr(file_path.size() - 4, 4) == ".csv") {
            data = CsvLoader::load_labels(file_path.c_str());
            break;
        }
    }
    for (int i = 0; i < data.size(); ++i) {
        data[i].first = path + "/" + data[i].first;
    }
}

Shape UnshuffledImgLoader::get_appropriate_shape(std::size_t index, std::size_t batch_size) const {
    auto img_size = ImageLoader::get_size(data[index].first.c_str());
    return Shape({batch_size, 3, img_size.first, img_size.second});
}

std::pair<std::vector<float>, float> UnshuffledImgLoader::get_raw(std::size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    std::string file_path = data[index].first;
    float ans = data[index].second;
    return {ImageLoader::load_image(file_path.c_str()), ans};
}

std::size_t UnshuffledImgLoader::size() const {
    return data.size();
}

void UnshuffledImgLoader::add_data(const UnshuffledDataLoader* other, int index) {
    data.push_back(reinterpret_cast<const UnshuffledImgLoader*>(other)->data[index]);
}