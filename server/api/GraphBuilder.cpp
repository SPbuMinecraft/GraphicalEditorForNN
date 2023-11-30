#include "GraphBuilder.h"
#include <vector>

void Graph::OverviewLayers(const crow::json::rvalue& layers, const std::vector<std::vector<float>> data,
                 std::unordered_map<int, crow::json::rvalue>& layer_dicts,
                 std::unordered_map<int, std::vector<float>>& data_dicts) {
    std::vector<float> instances;
    std::vector<float> answers;
    ParseCsvData(data, instances, answers);
    for (auto& layer : layers) {
        CHECK_HAS_FIELD(layer, "id");
        CHECK_HAS_FIELD(layer, "type");
        int id = static_cast<int>(layer["id"].i());
        std::string type = layer["type"].s();

        layerTypes_[id] = type;
        layer_dicts[id] = layer;

        if (type == "Data") {
            data_dicts[id] = instances; // All data goes for all data layers. May be should be splitted by them
            dataIds_.push_back(id);
        }
        if (type == "Target") {
            data_dicts[id] = answers;
        }
    }
}

void Graph::GetEdges(const crow::json::rvalue& connections,
              std::unordered_map<int, std::vector<int>>& straightEdges,
              std::unordered_map<int, std::vector<int>>& reversedEdges,
              std::unordered_set<int>& entryNodes) {
    std::unordered_set<int> nodesWithParents;
    for (auto& connection : connections) {
        if (!connection.has("layer_from") || !connection.has("layer_to")) {
            throw std::invalid_argument("Edge object has to contain both 'layer_from' and 'layer_to'");
        }
        int id_from = connection["layer_from"].i(), id_to = connection["layer_to"].i();
        // Straight edges
        straightEdges[id_from].push_back(id_to);
        // Reversed edges
        reversedEdges[id_to].push_back(id_from);

        // Loking for entries
        if (std::find(nodesWithParents.begin(), nodesWithParents.end(), id_from) ==
            nodesWithParents.end()) {
            entryNodes.insert(id_from);
        }
        if (std::find(nodesWithParents.begin(), nodesWithParents.end(), id_to) ==
            nodesWithParents.end()) {
            entryNodes.erase(id_to);
            nodesWithParents.insert(id_to);
        }
    }
}

void Graph::TopologySort(std::unordered_map<int, std::vector<int>>& edges,
                  std::unordered_set<int>& entryNodes,
                  std::vector<int>& layersOrder) {
    std::unordered_set<int> closed;
    std::stack<int> dfsStack;
    bool isFinal;

    for (int entryNode : entryNodes) {
        dfsStack.push(entryNode);
        while (!dfsStack.empty()) {
            int currentNode = dfsStack.top();
            isFinal = true;
            if (edges.find(currentNode) != edges.end()) {
                for (int nextNode : edges[currentNode]) {
                    if (std::find(closed.begin(), closed.end(), nextNode) != closed.end()) {
                        continue;
                    }
                    isFinal = false;
                    dfsStack.push(nextNode);
                }
            }
            if (isFinal) {
                layersOrder.push_back(currentNode);
                closed.insert(currentNode);
                dfsStack.pop();
            }
        }
    }

    std::reverse(layersOrder.begin(), layersOrder.end());
}

void Graph::Initialize(crow::json::rvalue modelJson,
             std::vector<std::vector<float>> data,
             RandomObject* randomInit,
             OptimizerBase& SGD) {

    CHECK_HAS_FIELD(modelJson, "graph");
    CHECK_HAS_FIELD(modelJson["graph"], "layers");
    CHECK_HAS_FIELD(modelJson["graph"], "connections");

    auto layersJson = modelJson["graph"]["layers"];
    auto edgesJson = modelJson["graph"]["connections"];

    // Parse Jsons into dicts of Jsons
    std::unordered_map<int, crow::json::rvalue> layerDicts;
    std::unordered_map<int, std::vector<float>> dataDicts;
    OverviewLayers(layersJson, data, layerDicts, dataDicts);

    std::unordered_map<int, std::vector<int>> straightEdges, reversedEdges;
    std::unordered_set<int> entryNodes;
    GetEdges(edgesJson, straightEdges, reversedEdges, entryNodes);

    std::vector<int> layersOrder;
    TopologySort(straightEdges, entryNodes, layersOrder);

    for (int layer_id : layersOrder) {
        std::vector<TensorRef> prevLayers;
        prevLayers.reserve(reversedEdges[layer_id].size());
        for (auto prevLayerId : reversedEdges[layer_id]) {
            prevLayers.push_back(layers_[prevLayerId]->result.value());
        }
        std::string type = layerTypes_.at(layer_id);
        if (type == "Linear") {
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            auto params = ParseLinear(layerDicts[layer_id]["parameters"]);
            layers_.emplace(layer_id, new LinearLayer{params, prevLayers, randomInit});
            SGD.append(layers_[layer_id]->layerOperationParams);
        } else if (type == "ReLU") {
            layers_.emplace(layer_id, new ReLULayer{prevLayers});
        } else if (type == "Data" || type == "Target") {
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            auto params = ParseData2d(layerDicts[layer_id]["parameters"]);
            if (dataDicts[layer_id].size() % params.width != 0) {
                throw std::invalid_argument("Sizes mismatch!");
            }
            params.height = dataDicts[layer_id].size() / params.width;
            layers_.emplace(layer_id, new Data2dLayer{params, dataDicts[layer_id]});
        } else if (type == "Output") {
            for (auto prevLayerId : reversedEdges[layer_id]) {
                lastPredictIds_.push_back(prevLayerId);
            }
        } else if (type == "MSELoss") {
            layers_.emplace(layer_id, new MSELoss{prevLayers});
            lastTrainIds_.push_back(layer_id);
        } else {
            throw std::invalid_argument("Unknown layer type: " + type);
        }
    }
}

void Graph::ChangeInputData(std::vector<float> data) {
    // All data goes to every data layer. Should be changed?
    for (int id : dataIds_) {
        Data2dLayer* layer = reinterpret_cast<Data2dLayer*>(layers_[id]);

        // Needs checks!!
        size_t width = layer->result->output->cols;
        if (data.size() % width != 0) {
            throw std::invalid_argument("Sizes mismatch!");
        }
        layer->result->output.emplace(Blob{data.size() / width, width, data.data()});
    }
}

Graph::~Graph() {
    for (auto it = layers_.begin(); it != layers_.end(); ++it) {
        delete it->second;
    }
}

std::vector<Layer*> Graph::getLastTrainLayers() const {
    std::vector<Layer*> result;
    result.reserve(lastTrainIds_.size());
    for (int id : lastTrainIds_) {
        result.push_back(layers_.at(id));
    }
    return result;
}

std::vector<Layer*> Graph::getLastPredictLayers() const {
    std::vector<Layer*> result;
    result.reserve(lastPredictIds_.size());
    for (int id : lastPredictIds_) {
        result.push_back(layers_.at(id));
    }
    return result;
}
const Layer& Graph::operator[](int i) const {
    return *layers_.at(i);
}
