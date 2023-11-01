#include <iostream>
#include <vector>
#include <string>

#include <crow_all.h>
#include "GraphBuilder.h"

using namespace std;
using namespace crow;


float train(json::rvalue json, Graph** graph) {
    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);
    *graph = new Graph(json, &initObject, SGD);
    std::cout << "Graph is ready!" << std::endl;

    Blob result {1, 1};

    auto lastNode = (*graph)->getLastTrainLayers()[0]->result.value();  // Пока не думаем о нескольких выходах (!) Hard-coded
    for (int j = 0; j < 1000; ++j) {
        result = lastNode.forward();
        printf("%d: %f\n", j, result[0][0]);
        // lastNode.gradient = result;
        lastNode.gradient.value()[0][0] = 1;
        lastNode.backward();
        SGD.step();
        lastNode.clear();
    }

    return 0.0f;
}

float predict(json::rvalue json, Graph* graph, std::vector<json::wvalue>& answer) {
    graph->ChangeInputData(json);

    auto lastNode = graph->getLastPredictLayers()[0]->result.value();  // Пока не думаем о нескольких выходах (!) Hard-coded
    lastNode.clear();
    Blob result = lastNode.forward();
    lastNode.clear();
    
    for (size_t j = 0; j < result.rows; ++j) {
        for (size_t i = 0; i < result.cols; ++i) {
            answer.push_back(json::wvalue(static_cast<float>(result[j][i])));
            std::cout << result[j][i] << std::endl;
        }
    }
    return 0.0f;
}

long loadConfig() {
    ifstream t("../config.json");
    stringstream buffer;
    buffer << t.rdbuf();
    auto configs = json::load(buffer.str());
    return configs["cpp_server"]["PORT"].i();
}


int main() {
    SimpleApp app;
    auto port = loadConfig();

    //Graph* graph = nullptr;  // Doesn't work, needs sessions

    std::map<int, Graph*> sessions;

    CROW_ROUTE(app, "/predict/<int>").methods(HTTPMethod::POST)
    ([&](const request& req, int model_id) -> response {
        auto body = json::load(req.body);
        if (!body) return response(status::BAD_REQUEST, "Invalid body");
        if (sessions.find(model_id) == sessions.end()) return response(status::BAD_REQUEST, "Not trained");
        std::vector<json::wvalue> answer;
        try {
            predict(body, sessions[model_id], answer);
        } catch (const std::runtime_error &err) {
            return response(status::BAD_REQUEST, "Invalid body");
        }
        json::wvalue response = {{"answer", answer}};
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
        sessions[model_id] = nullptr;
        train(body, &sessions[model_id]);
        return response(status::OK, "done");
    });

    app.port(port).multithreaded().run();

    for (auto model_graph: sessions) {
        delete model_graph.second;
    }
    
    return 0;
}
