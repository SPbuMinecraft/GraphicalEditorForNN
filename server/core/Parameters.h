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

struct AxisParameters {
    std::vector<short> axis;
};

struct CrossEntropyLossParameters {
    std::size_t classCount; 
};
