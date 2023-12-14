#pragma once

#define cimg_use_png 1
#include "CImg.h" // required "sudo apt install libx11-dev"
#include <string>
#include <vector>

class ImageLoader {
public:
    static std::vector<float> load_image(const char* path);
    static std::vector<float> get_pixels(cimg_library::CImg<unsigned char>);
    static std::pair<std::size_t, std::size_t> get_size(const char* path);
};
