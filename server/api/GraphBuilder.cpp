#include "GraphBuilder.h"
#include "Allocator.h"

void Graph::OverviewLayers(const crow::json::rvalue& layers, 
                           std::unordered_map<int, crow::json::rvalue>& layer_dicts) {
    for (auto& layer : layers) {
        CHECK_HAS_FIELD(layer, "id");
        CHECK_HAS_FIELD(layer, "type");
        int id = static_cast<int>(layer["id"].i());
        std::string type = layer["type"].s();

        layerTypes_[id] = type;
        layer_dicts[id] = layer;
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
                       RandomObject* randomInit,
                       OptimizerBase& SGD,
                       size_t batch_size) {
    Allocator::startVirtualMode();
    CHECK_HAS_FIELD(modelJson, "graph");
    CHECK_HAS_FIELD(modelJson["graph"], "layers");
    CHECK_HAS_FIELD(modelJson["graph"], "connections");

    auto layersJson = modelJson["graph"]["layers"];
    auto edgesJson = modelJson["graph"]["connections"];

    // Parse Jsons into dicts of Jsons
    std::unordered_map<int, crow::json::rvalue> layerDicts;
    OverviewLayers(layersJson, layerDicts);

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
            if (type == "Data") {
                dataIds_.push_back(layer_id);
            }
            if (type == "Target") {
                targetsIds_.push_back(layer_id);
            }
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            Shape shape = ParseData(layerDicts[layer_id]["parameters"]);
            layers_.emplace(layer_id, new DataLayer{shape, batch_size});
        } else if (type == "Output") {
            for (auto prevLayerId : reversedEdges[layer_id]) {
                lastPredictIds_.push_back(prevLayerId);
            }
        } else if (type == "MSELoss") {
            layers_.emplace(layer_id, new MSELoss{prevLayers});
            lastTrainIds_.push_back(layer_id);
        } else if (type == "CrossEntropyLoss") {
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            auto params = ParseCrossEntropyLoss(layerDicts[layer_id]["parameters"]);
            layers_.emplace(layer_id, new EntropyLoss{params, prevLayers});
            lastTrainIds_.push_back(layer_id);
        } else if (type == "Conv2d") {
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            auto params = ParseConv2d(layerDicts[layer_id]["parameters"]);
            layers_.emplace(layer_id, new Conv2DLayer{params, prevLayers, randomInit});
            SGD.append(layers_[layer_id]->layerOperationParams);
        } else if (type == "LayerNorm") {
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            auto params = ParseAxes(layerDicts[layer_id]["parameters"]);
            layers_.emplace(layer_id, new LayerNorm{params, prevLayers});
        } else if (type == "MaxPool") {
            layers_.emplace(layer_id, new MaxPool{prevLayers});
        } else if (type == "SoftMax") {
            CHECK_HAS_FIELD(layerDicts[layer_id], "parameters");
            auto params = ParseAxes(layerDicts[layer_id]["parameters"]);
            layers_.emplace(layer_id, new SoftMax{params, prevLayers});
        } else {
            throw std::invalid_argument("Unknown layer type: " + type);
        }
    }
}

void Graph::ChangeLayersData(std::vector<float> data, BaseLayerType type) {
    // All data goes to every data layer. Should be changed?
    std::vector<int>* layers = nullptr;
    if (type == BaseLayerType::Data) {
        layers = &dataIds_;
    } else if (type == BaseLayerType::Targets) {
        layers = &targetsIds_;
    } else {
        throw std::invalid_argument("Can change data only in 'Data' or 'Target' layers");
    }
    for (int id : *layers) {
        DataLayer* layer = reinterpret_cast<DataLayer*>(layers_[id]);

        Shape expected_shape = layer->result->output->shape;
        size_t sample_size = expected_shape.stride(4 - expected_shape.dimsCount);

        if (data.size() % sample_size != 0 || data.size() > expected_shape.size()) {
            std::cerr << data.size() << " " << sample_size << " " << expected_shape.size() << std::endl;
            throw std::invalid_argument("Sizes mismatch!");
        }
        data.resize(expected_shape.size(), 0);
        layer->result->output.emplace(Blob::constBlob(expected_shape, data.data()));
    }
}

Graph::~Graph() {
    for (auto it = layers_.begin(); it != layers_.end(); ++it) {
        delete it->second;
    }
    Allocator::end();
}

std::vector<Layer*> Graph::getLayers(BaseLayerType type) const {
    std::vector<Layer*> result;
    std::vector<int>* layers_ids = nullptr;
    if (type == BaseLayerType::Data) {
        layers_ids = const_cast<std::vector<int>*>(&dataIds_);
    } else if (type == BaseLayerType::Targets) {
        layers_ids = const_cast<std::vector<int>*>(&targetsIds_);
    } else if (type == BaseLayerType::TrainOut) {
        layers_ids = const_cast<std::vector<int>*>(&lastTrainIds_);
    } else {
        layers_ids = const_cast<std::vector<int>*>(&lastPredictIds_);
    }

    result.reserve(layers_ids->size());
    for (int id : *layers_ids) {
        result.push_back(layers_.at(id));
    }
    return result;
}

const Layer& Graph::operator[](int i) const {
    return *layers_.at(i);
}
