#include "Parser.h"

void CHECK_HAS_FIELD(const crow::json::rvalue& layer, const std::string& field) {
    if (!layer.has(field)) {
        throw std::invalid_argument("Field '" + field + "' is required but not found");
    }
}

// void ParseCsvData(const std::vector<std::vector<float>>& data, std::vector<float>& instances, std::vector<float>& answers) {
//     instances.reserve(data.size());
//     answers.reserve(data.size());
//     for (auto& instance : data) {
//         answers.push_back(instance.back());
//         instances.emplace_back(instance.begin(), std::prev(instance.end()));
//     }
// }

LinearLayerParameters ParseLinear(const crow::json::rvalue& parameters) {
    size_t inFeatures, outFeatures;
    bool bias = true;

    CHECK_HAS_FIELD(parameters, "inFeatures");
    CHECK_HAS_FIELD(parameters, "outFeatures");

    inFeatures = static_cast<size_t>(parameters["inFeatures"].i());
    outFeatures = static_cast<size_t>(parameters["outFeatures"].i());

    // if (parameters.has("bias")) {
    //     std::cout << parameters["bias"] << std::endl;
    //     bias = parameters["bias"].b();
    //     std::cout << "done" << std::endl;
    // }

    return LinearLayerParameters{inFeatures, outFeatures, bias};
}

Shape ParseData(const crow::json::rvalue& parameters) {
    std::vector<size_t> shape;
    shape.reserve(3);

    CHECK_HAS_FIELD(parameters, "shape");

    for (auto dim : parameters["shape"]) {
        shape.push_back(dim.i());
    }
    return Shape{std::move(shape)};
}
