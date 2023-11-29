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

void ParseInputData(const crow::json::rvalue& data, std::vector<float>& result);

LinearLayerParameters ParseLinear(const crow::json::rvalue& parameters);
Data2dLayerParameters ParseData2d(const crow::json::rvalue& parameters);
