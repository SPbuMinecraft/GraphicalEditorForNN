#include <set>
#include <vector>

#include <crow_all.h>
#include "Parser.h"
#include "Layer.h"

class Graph {
private:
    std::unordered_map<int, Layer*> layers_ = {};
    std::unordered_map<int, std::string> layerTypes_ = {};
    std::vector<int> lastTrainIds_ = {};
    std::vector<int> lastPredictIds_ = {};
    std::vector<int> dataIds_ = {};

public:
    Graph() = default;
    void Initialize(crow::json::rvalue modelJson,
          std::vector<std::vector<float>> data,
          RandomObject* randomInit,
          OptimizerBase& SGD);
    ~Graph();

    void OverviewLayers(const crow::json::rvalue& layers, const std::vector<std::vector<float>>& data,
                        std::unordered_map<int, crow::json::rvalue>& layer_dicts,
                        std::unordered_map<int, std::vector<float>>& data_dicts);

    void GetEdges(const crow::json::rvalue& connections,
                  std::unordered_map<int, std::vector<int>>& straightEdges,
                  std::unordered_map<int, std::vector<int>>& reversedEdges,
                  std::unordered_set<int>& entryNodes);

    void TopologySort(std::unordered_map<int, std::vector<int>>& edges,
                      std::unordered_set<int>& entryNodes,
                      std::vector<int>& layersOrder);

    void ChangeInputData(std::vector<float> data);
    
    std::vector<Layer*> getLastTrainLayers() const;
    std::vector<Layer*> getLastPredictLayers() const;

    const Layer& operator[](int i) const;
};
