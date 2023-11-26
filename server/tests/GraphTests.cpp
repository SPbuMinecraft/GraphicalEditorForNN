#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <fstream>
#include <sstream>

#include "doctest.h"

#include "GraphBuilder.h"

using Pair = std::pair<int, int>;


void CheckFullProcess(const std::string& json_path,
                      const std::vector<int>& layer_ids_expected,
                      const std::vector<int>& data_ids_expected,
                      const std::vector<Pair>& edges_expected,
                      int edges_storage_size_expected,
                      const std::vector<int>& entries_expected,
                      const std::vector<std::vector<bool>>& less_matrix_expected) {
    Graph g;

    std::ifstream input("./tests/" + json_path);
    if (!input) {
        throw std::runtime_error("No file named 'example.json'");
    }
    std::ostringstream ss;
    ss << input.rdbuf();
    crow::json::rvalue json = crow::json::load(ss.str().data(), ss.str().length());

    auto layersJson = json["graph"]["layers"];
    auto edgesJson = json["graph"]["connections"];
    auto dataJson = json["dataset"];

    // #################################################################
    //                     PARSE LAYERS IDS
    // #################################################################
    std::unordered_map<int, crow::json::rvalue> layerDicts, dataDicts;
    g.OverviewLayers(layersJson, dataJson, layerDicts, dataDicts);

    CHECK(layerDicts.size() == layer_ids_expected.size());
    for (auto id : layer_ids_expected) {
        CHECK(layerDicts.find(id) != layerDicts.end());
    }

    CHECK(dataDicts.size() == data_ids_expected.size());
    for (auto id : data_ids_expected) {
        CHECK(dataDicts.find(id) != dataDicts.end());
    }

    // #################################################################
    //                         PARSE EDGES
    // #################################################################
    std::unordered_map<int, std::vector<int>> straightEdges, reversedEdges;
    std::unordered_set<int> entryNodes;
    g.GetEdges(edgesJson, straightEdges, reversedEdges, entryNodes);


    CHECK(straightEdges.size() == edges_storage_size_expected);
    CHECK(reversedEdges.size() == edges_storage_size_expected);
    for (auto& [id_from ,id_to] : edges_expected) {
        CHECK(straightEdges.find(id_from) != straightEdges.end());
        CHECK(std::find(straightEdges[id_from].begin(), straightEdges[id_from].end(), id_to) !=
              straightEdges[id_from].end());
        CHECK(reversedEdges.find(id_to) != reversedEdges.end());
        CHECK(std::find(reversedEdges[id_to].begin(), reversedEdges[id_to].end(), id_from) !=
              reversedEdges[id_to].end());
    }

    CHECK(entryNodes.size() == entries_expected.size());
    for (int id : entries_expected) {
        CHECK(entryNodes.find(id) != entryNodes.end());
    }

    // #################################################################
    //                         TOPOLOGY SORT
    // #################################################################
    std::vector<int> layersOrder;
    g.TopologySort(straightEdges, entryNodes, layersOrder);

    // Здесь надо придумать более адекватный по памяти способ проверить
    // топологическую сортировку
    CHECK(layersOrder.size() == layer_ids_expected.size());
    for (size_t index = 0; index < layersOrder.size() - 1; ++index) {
        CHECK(!less_matrix_expected[layersOrder[index + 1]][layersOrder[index]]);
    }
}

TEST_CASE("Linear-ReLU-Linear-MSE") {
    std::vector<int> layer_ids_expected = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int> data_ids_expected = {0, 4};
    std::vector<Pair> edges_expected = {
        {0, 1}, {1, 2}, {2, 3}, {3, 5}, {4, 5}, {3, 6}
    };
    int edges_storage_size_expected = 5;
    std::vector<int> entries_expected = {0, 4};
    std::vector<std::vector<bool>> less_matrix_expected = {
        {0, 1, 1, 1, 0, 1, 1},
        {0, 0, 1, 1, 0, 1, 1},
        {0, 0, 0, 1, 0, 1, 1},
        {0, 0, 0, 0, 0, 1, 1},
        {0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0}
    };

    CheckFullProcess("linear_relu_linear_mse.json", layer_ids_expected, data_ids_expected,
                     edges_expected, edges_storage_size_expected,
                     entries_expected, less_matrix_expected);
}