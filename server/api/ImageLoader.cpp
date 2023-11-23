#include "ImageLoader.h"
#include <iostream>

std::vector<float> ImageLoader::load_image(char* path) {
    cimg_library::CImg<unsigned char> image(path);
    return get_pixels(image);
}

std::vector<float> ImageLoader::get_pixels(cimg_library::CImg<unsigned char> img) {
    const int colorsCnt = 3;
    int size = img.width() * img.height() * colorsCnt;
    std::vector<float> ans(size);
    std::cout << size << std::endl;
    for (int i = 0; i < img.width(); ++i) {
        for (int j = 0; j < img.height(); ++j) {
            for (int k = 0; k < colorsCnt; ++k) {
                int channel = k;
                if (channel >= img.spectrum()) {
                    channel = 0;
                }
                int ans_index = i * img.height() * colorsCnt + j * colorsCnt + k;
                ans[ans_index] = img(i, j, 0, channel);
            }
        }
    }
    std::cout << std::endl;
    return ans;
}