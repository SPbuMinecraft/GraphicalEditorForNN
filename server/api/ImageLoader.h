#pragma once

#include "CImg.h" // required "sudo apt install libx11-dev"
#include <string>
#include <vector>

class ImageLoader {
public:
    std::vector<float> load_image(char* path);
    std::vector<float> get_pixels(cimg_library::CImg<unsigned char>);
};
