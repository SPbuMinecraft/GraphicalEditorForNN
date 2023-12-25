#include "Parser.h"

void CHECK_HAS_FIELD(const crow::json::rvalue& layer, const std::string& field) {
    if (!layer.has(field)) {
        throw std::invalid_argument("Field '" + field + "' is required but not found");
    }
}

LinearLayerParameters ParseLinear(const crow::json::rvalue& parameters) {
    size_t inFeatures, outFeatures;
    bool bias = true;

    CHECK_HAS_FIELD(parameters, "inFeatures");
    CHECK_HAS_FIELD(parameters, "outFeatures");

    inFeatures = static_cast<size_t>(parameters["inFeatures"].i());
    outFeatures = static_cast<size_t>(parameters["outFeatures"].i());

    // Something fails here, TODO: catch the bug
    if (parameters.has("bias")) {
        bias = static_cast<bool>(parameters["bias"].i());
        std::cout << "done" << std::endl;
    }

    return LinearLayerParameters{inFeatures, outFeatures, bias};
}

Shape ParseData(const crow::json::rvalue& parameters) {
    CHECK_HAS_FIELD(parameters, "shape");

    std::vector<size_t> shape;
    shape.reserve(3);

    for (auto dim : parameters["shape"]) {
        shape.push_back(dim.i());
    }
    return Shape{std::move(shape)};
}

Conv2DLayerParameters ParseConv2d(const crow::json::rvalue& parameters) {
    CHECK_HAS_FIELD(parameters, "kernelSize");
    CHECK_HAS_FIELD(parameters, "inChannels");
    CHECK_HAS_FIELD(parameters, "outChannels");

    size_t kernelSize, inChannels, outChannels;
    kernelSize = static_cast<size_t>(parameters["kernelSize"].i());
    inChannels = static_cast<size_t>(parameters["inChannels"].i());
    outChannels = static_cast<size_t>(parameters["outChannels"].i());

    return Conv2DLayerParameters{kernelSize, inChannels, outChannels};
}

AxisParameters ParseAxes(const crow::json::rvalue& parameters) {
    CHECK_HAS_FIELD(parameters, "axes");

    std::vector<short> axes;
    for (auto ax_num : parameters["axes"]) {
        axes.push_back(static_cast<short>(ax_num.i()));
    }
    return AxisParameters{std::move(axes)};
}

CrossEntropyLossParameters ParseCrossEntropyLoss(const crow::json::rvalue& parameters) {
    CHECK_HAS_FIELD(parameters, "classCount");
    return CrossEntropyLossParameters{static_cast<size_t>(parameters["classCount"].i())};
}
