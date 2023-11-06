#include "Parser.h"

void CHECK_HAS_FIELD(const crow::json::rvalue& layer, const std::string& field) {
    if (!layer.has(field)) {
        throw std::invalid_argument("Field '" + field + "' is required but not found");
    }
}

void ParseInputData(const crow::json::rvalue& data, std::vector<float>& result) {
    // Think about optimization via reservation
    for (auto& inputValue : data) {
        result.push_back(static_cast<float>(inputValue.d()));
    }
}

LinearLayerParameters ParseLinear(const crow::json::rvalue& parameters) {
    size_t inFeatures, outFeatures;
    bool bias = true;

    CHECK_HAS_FIELD(parameters, "inFeatures");
    CHECK_HAS_FIELD(parameters, "outFeatures");

    inFeatures = static_cast<size_t>(parameters["inFeatures"].i());
    outFeatures = static_cast<size_t>(parameters["outFeatures"].i());

    if (parameters.has("bias")) {
        bias = parameters["bias"].b();
    }

    return LinearLayerParameters{inFeatures, outFeatures, bias};
}

Data2dLayerParameters ParseData2d(const crow::json::rvalue& parameters) {
    size_t width;

    CHECK_HAS_FIELD(parameters, "width");

    width = static_cast<size_t>(parameters["width"].i());
    return Data2dLayerParameters{.width = width};
}
