#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <crow_all.h>

#include "Tensor.h"
#include "RandomInit.h"
#include "Optimizer.h"
#include "Parameters.h"


void CHECK_HAS_FIELD(const crow::json::rvalue& layer, const std::string& field);

// void ParseCsvData(const std::vector<std::vector<float>>& data, std::vector<float>& instances, std::vector<float>& answers);

LinearLayerParameters ParseLinear(const crow::json::rvalue& parameters);
Conv2DLayerParameters ParseConv2d(const crow::json::rvalue& parameters);
AxisParameters ParseAxes(const crow::json::rvalue& parameters);
CrossEntropyLossParameters ParseCrossEntropyLoss(const crow::json::rvalue& parameters);

Shape ParseData(const crow::json::rvalue& parameters);
