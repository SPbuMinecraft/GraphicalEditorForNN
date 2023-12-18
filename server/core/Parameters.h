#pragma once

#include <vector>

struct LinearLayerParameters {
    std::size_t inFeatures;
    std::size_t outFeatures;
    bool bias;
};

struct Conv2DLayerParameters {
    std::size_t kernelSize;
    std::size_t inChannels;
    std::size_t outChannels;
};

struct Data2dLayerParameters {
    std::size_t width;
    std::size_t height;
};

struct AxisParameters
{
    std::vector<short> axis;
};

