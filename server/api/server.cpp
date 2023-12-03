#include <iostream>
#include <vector>
#include <string>

#include <crow_all.h>
#include "GraphBuilder.h"

using namespace std;
using namespace crow;


void train(json::rvalue& json, Graph** graph) {
    RandomObject initObject(0, 1, 17);
    OptimizerBase SGD = OptimizerBase(0.1);
    *graph = new Graph();
    (*graph)->Initialize(json, &initObject, SGD);
    std::cout << "Graph is ready!" << std::endl;

    auto& lastNode = (*graph)->getLastTrainLayers()[0]->result.value();  // Пока не думаем о нескольких выходах (!) Hard-coded

    lastNode.forward();
    lastNode.gradient = Blob::ones({{1}});
    lastNode.backward();
    Allocator::endSession();
    lastNode.clear();
    Allocator::endVirtualMode();

    for (int j = 0; j < 1000; ++j) {
        auto& result = lastNode.forward();
        printf("%d: %f\n", j, result(0, 0, 0, 0));
        // lastNode.gradient = result;
        lastNode.gradient = Blob::ones({{1}});
        lastNode.backward();
        SGD.step();
        lastNode.clear();
    }
}

void predict(json::rvalue& json, Graph* graph, std::vector<float>& answer) {
    graph->ChangeInputData(json);

    auto& lastNode = graph->getLastPredictLayers()[0]->result.value();  // Пока не думаем о нескольких выходах (!) Hard-coded
    lastNode.clear();
    const Blob& result = lastNode.forward();

    answer.reserve(result.shape.rows() * result.shape.cols());
    for (size_t j = 0; j < result.shape.rows(); ++j) {
        for (size_t i = 0; i < result.shape.cols(); ++i) {
            answer.push_back(result(j, i));
            std::cout << result(j, i) << std::endl;
        }
    }
}

void invalidArgs() { 
    cout << "Usage: ./server <host: str> <port: int>" << endl;
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc != 3) invalidArgs();
    SimpleApp app;

    std::map<int, Graph*> sessions;

    CROW_ROUTE(app, "/predict/<int>").methods(HTTPMethod::POST)
    ([&](const request& req, int model_id) -> response {
        auto body = json::load(req.body);
        if (!body) return response(status::BAD_REQUEST, "No model provided");
        if (sessions.find(model_id) == sessions.end()) return response(status::METHOD_NOT_ALLOWED, "Not trained");
        std::vector<float> answer;
        try {
            predict(body, sessions[model_id], answer);
        } catch (const std::runtime_error &err) {
            return response(status::BAD_REQUEST, "Invalid body");
        }
        json::wvalue response;
        for (int i = 0; i < answer.size(); ++i) {
            response[i] = answer[i];
        }
        return crow::response(status::OK, response);
    });

    CROW_ROUTE(app, "/train/<int>").methods(HTTPMethod::POST)
    ([&](const request& req, int model_id) -> response {
        auto body = json::load(req.body);
        std::cout << "Checking json!" << std::endl;
        if (!body) return response(status::BAD_REQUEST, "Invalid body");
        std::cout << "Training" << std::endl;
        if (sessions.find(model_id) != sessions.end() && sessions[model_id] != nullptr) {
            delete sessions[model_id];
        }
        Graph* g = nullptr;
        train(body, &g);
        sessions[model_id] = g;
        return response(status::OK, "done");
    });

    int port;
    if (sscanf(argv[2], "%d", &port) != 1) invalidArgs();

    // for now we cannot handle even one thread (
    app.port(port).run();


    for (auto model_graph: sessions) {
        delete model_graph.second;
    }
    
    return 0;
}
