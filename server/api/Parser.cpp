#include "Parser.h"

void CHECK_HAS_FIELD(const crow::json::rvalue& layer, const std::string& field) {
    if (!layer.has(field)) {
        throw std::invalid_argument("Field '" + field + "' is required but not found");
    }
}

LinearLayerParameters ParseLinear(crow::json::rvalue parameters) {
    size_t inFeatures, outFeatures;
    bool bias = true;
    std::string device = "cpu";

    CHECK_HAS_FIELD(parameters, "inFeatures");
    CHECK_HAS_FIELD(parameters, "outFeatures");

    inFeatures = static_cast<size_t>(parameters["inFeatures"].i());
    outFeatures = static_cast<size_t>(parameters["outFeatures"].i());

    if (parameters.has("bias")) {
        bias = parameters["bias"].b();
    }
    if (parameters.has("device")) {
        device = parameters["device"].s();
    }

    return LinearLayerParameters{inFeatures, outFeatures, bias, device};
}

Data2dLayerParameters ParseData2d(crow::json::rvalue parameters) {
    size_t width, height;

    CHECK_HAS_FIELD(parameters, "width");
    CHECK_HAS_FIELD(parameters, "height");

    width = static_cast<size_t>(parameters["width"].i());
    height = static_cast<size_t>(parameters["height"].i());

    return Data2dLayerParameters{width, height};
}
