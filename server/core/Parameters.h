#pragma once

struct LinearLayerParameters {
    std::size_t inFeatures;
    std::size_t outFeatures;
    bool bias;
};

struct Data2dLayerParameters {
    std::size_t width;
    std::size_t height;
};
