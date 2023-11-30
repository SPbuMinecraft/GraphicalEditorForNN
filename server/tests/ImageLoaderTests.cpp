#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "ImageLoader.h"
#include <vector>

TEST_CASE("one_pixel") {
    std::vector<float> white_pixel = {255, 255, 255};
    std::vector<float> black_pixel = {0, 0, 0};
    std::vector<float> lazure_pixel = {159, 252, 253};

    SUBCASE("white_pixel") {
        auto result = ImageLoader::load_image("tests/data/white_pixel.png");
        CHECK(result.size() == white_pixel.size());
        for (int i = 0; i < white_pixel.size(); ++i) {
            CHECK(white_pixel[i] == result[i]);
        }
    }

    SUBCASE("black_pixel") {
        auto result = ImageLoader::load_image("tests/data/black_pixel.png");
        CHECK(result.size() == black_pixel.size());
        for (int i = 0; i < black_pixel.size(); ++i) {
            CHECK(black_pixel[i] == result[i]);
        }
    }

    SUBCASE("lazure_pixel") {
        auto result = ImageLoader::load_image("tests/data/lazure_pixel.png");
        CHECK(result.size() == lazure_pixel.size());
        for (int i = 0; i < lazure_pixel.size(); ++i) {
            CHECK(lazure_pixel[i] == result[i]);
        }
    }
}

TEST_CASE("big_picture") {
    std::vector<float> traffic_light = {255, 255, 0, 0, 255, 255, 0, 0, 0};
    std::vector<float> picture = {0, 255, 100, 153, 136, 255, 0, 174, 100, 217, 0, 255, 0, 201, 100, 234, 21, 255};

    SUBCASE("traffic_light") {
        auto result = ImageLoader::load_image("tests/data/traffic_light.png");
        CHECK(result.size() == traffic_light.size());
        for (int i = 0; i < traffic_light.size(); ++i) {
            CHECK(traffic_light[i] == result[i]);
        }
    }

    SUBCASE("random_picture") {
        auto result = ImageLoader::load_image("tests/data/picture.png");
        CHECK(result.size() == picture.size());
        for (int i = 0; i < picture.size(); ++i) {
            CHECK(picture[i] == result[i]);
        }
    }
}