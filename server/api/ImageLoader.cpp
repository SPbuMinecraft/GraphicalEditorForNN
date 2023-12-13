#include "ImageLoader.h"

std::vector<float> ImageLoader::load_image(const char* path) {
    cimg_library::CImg<unsigned char> image(path);
    return get_pixels(image);
}

std::vector<float> ImageLoader::get_pixels(cimg_library::CImg<unsigned char> img) {
    const int colorsCnt = 3;
    int size = img.width() * img.height() * colorsCnt;
    std::vector<float> ans(size);
    for (int k = 0; k < colorsCnt; ++k) {
        for (int i = 0; i < img.width(); ++i) {
            for (int j = 0; j < img.height(); ++j) {
                int channel = k;
                if (channel >= img.spectrum()) {
                    channel = 0;
                }
                int ans_index = k * img.height() * img.width() + i * img.height() + j;
                ans[ans_index] = img(i, j, 0, channel);
            }
        }
    }
    return ans;
}

std::pair<std::size_t, std::size_t> ImageLoader::get_size(const char *path) {
    cimg_library::CImg<unsigned char> image(path);
    return {image.width(), image.height()};
}