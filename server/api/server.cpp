#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <crow_all.h>
#include "GraphBuilder.h"
#include "CsvLoader.h"

using namespace std;
using namespace crow;

std::string getDataPath(int id) {
    return "./model_data/data/" + std::to_string(id) + ".csv";
}

std::string getPredictPath(int id) {
    return "./model_data/predict/" + std::to_string(id) + ".csv";
}

void train(json::rvalue& json, Graph** graph, int model_id) {
    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);
    std::vector<std::vector<float>> data = CsvLoader::load_csv(getDataPath(model_id));
    *graph = new Graph();
    (*graph)->Initialize(json, data, &initObject, SGD);
    std::cout << "Graph is ready!" << std::endl;

    Blob result {1, 1};

    auto& lastNode = (*graph)->getLastTrainLayers()[0]->result.value();  // Пока не думаем о нескольких выходах (!) Hard-coded
    for (int j = 0; j < 1000; ++j) {
        result = lastNode.forward();
        printf("%d: %f\n", j, result[0][0]);
        // lastNode.gradient = result;
        lastNode.gradient.value()[0][0] = 1;
        lastNode.backward();
        SGD.step();
        lastNode.clear();
    }
}

void predict(int model_id, Graph* graph, std::vector<float>& answer) {
    std::vector<std::vector<float>> predict_data = CsvLoader::load_csv(getPredictPath(model_id));
    graph->ChangeInputData(predict_data[0]);

    auto& lastNode = graph->getLastPredictLayers()[0]->result.value();  // Пока не думаем о нескольких выходах (!) Hard-coded
    lastNode.clear();
    Blob result = lastNode.forward();
    lastNode.clear();
    
    answer.reserve(result.rows * result.cols);
    for (size_t j = 0; j < result.rows; ++j) {
        for (size_t i = 0; i < result.cols; ++i) {
            answer.push_back(result[j][i]);
            std::cout << result[j][i] << std::endl;
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
        if (sessions.find(model_id) == sessions.end()) return response(status::METHOD_NOT_ALLOWED, "Not trained");
        std::vector<float> answer;
        try {
            predict(model_id, sessions[model_id], answer);
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
        train(body, &g, model_id);
        sessions[model_id] = g;
        return response(status::OK, "done");
    });

    //curl -X POST -F "InputFile=@/home/osboxes/CLionProjects/GraphicalEditorForNN/server/tests/data/and-train.csv" http://0.0.0.0:2000/upload_data/1
    CROW_ROUTE(app, "/upload_data/<int>/<int>").methods(HTTPMethod::Post)
    ([&](const request& req, int model_id, int type) -> response {
        crow::multipart::message file_message(req);
        std::string path;
        if (type == 0) {
            path = getDataPath(model_id);
        }
        else {
            path = getPredictPath(model_id);
        }
        std::ofstream out_file(path);
        if (!out_file) {
            return response(status::INTERNAL_SERVER_ERROR, "Failed to open file for storage");
        }
        auto content = file_message.part_map.find("InputFile");
        if (content == file_message.part_map.end()) {
            return response(status::BAD_REQUEST, "No file provided");
        }
        out_file << (*content).second.body;
        out_file.close();
        return crow::response(status::OK, "done");
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
