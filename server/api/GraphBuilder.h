#include <set>

#include <crow_all.h>
#include "Layer.h"

class Graph {
private:
    std::unordered_map<int, Layer*> layers_ = {};
    std::unordered_map<int, std::string> layerTypes_ = {};
    std::vector<int> lastTrainIds_ = {};
    std::vector<int> lastPredictIds_ = {};

    void OverviewLayers(const crow::json::rvalue& layers, const crow::json::rvalue& data,
                        std::unordered_map<int, crow::json::rvalue>& layer_dicts,
                        std::unordered_map<int, crow::json::rvalue>& data_dicts);
    void GetEdges(const crow::json::rvalue& connections,
                  std::unordered_map<int, std::vector<int>>& straightEdges,
                  std::unordered_map<int, std::vector<int>>& reversedEdges,
                  std::unordered_set<int>& entryNodes);
    void TopologySort(std::unordered_map<int, std::vector<int>>& edges,
                      std::unordered_set<int>& entryNodes,
                      std::vector<int>& layersOrder);

public:
    Graph() = default;
    Graph(crow::json::rvalue modelJson,
          RandomObject* randomInit,
          OptimizerBase& SGD);
    ~Graph();

    void ChangeInputData(crow::json::rvalue& data);
    
    std::vector<Layer*> getLastTrainLayers() const;
    std::vector<Layer*> getLastPredictLayers() const;

    const Layer& operator[](int i) const;
};