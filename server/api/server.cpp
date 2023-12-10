#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <boost/asio.hpp>

#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
#include <cpprest/uri.h>
#include <cpprest/json.h>
#include <filesystem>

#include <crow_all.h>
#include "zip.h"
#include "GraphBuilder.h"
#include "CsvLoader.h"

using namespace std;
using namespace crow;

std::string getDataPath(int id) {
    return "./model_data/data/" + std::to_string(id);
}

std::string getPredictPath(int id) {
    return "./model_data/predict/" + std::to_string(id);
}

web::json::value GetLogs(const std::optional<Blob>& node) {
    assert(node.has_value() && node.value().shape.dimsCount <= 2);
    std::vector<web::json::value> values;
    values.reserve(node.value().shape.size());
    for (size_t sample_index = 0; sample_index < node.value().shape.rows(); ++sample_index) {
        for (size_t feature_index = 0; feature_index < node.value().shape.cols(); ++feature_index) {
            values.push_back(web::json::value::number(node.value()(0, 0, sample_index, feature_index)));
        }
    }
    return web::json::value::array(values);
}

void train(json::rvalue& json, Graph** graph, int model_id, int user_id) {
    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);
    std::vector<std::vector<float>> data = CsvLoader::load_csv(getDataPath(model_id));
    *graph = new Graph();
    (*graph)->Initialize(json, data, &initObject, SGD);
    std::cout << "Graph is ready!" << std::endl;

    auto& lastTrainNode = (*graph)->getLayers(BaseLayerType::TrainOut)[0]->result.value();
    auto& lastPredictNode = (*graph)->getLayers(BaseLayerType::PredictOut)[0]->result.value().output;
    auto& targetsNode = (*graph)->getLayers(BaseLayerType::Targets)[0]->result.value().output;

    lastTrainNode.forward();
    lastTrainNode.gradient = Blob::ones({{1}});
    lastTrainNode.backward();
    Allocator::endSession();
    lastTrainNode.clear();
    Allocator::endVirtualMode();

    size_t buffer_size = 5;
    std::vector<web::json::value> targets, outputs;
    targets.reserve(buffer_size);
    outputs.reserve(buffer_size);

    size_t max_epochs = 1000;
    for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
        auto& result = lastTrainNode.forward();
        // printf("%ld: %f\n", epoch, result[0][0]);

        outputs.push_back(GetLogs(lastPredictNode));
        targets.push_back(GetLogs(targetsNode));

        if ((epoch == max_epochs - 1 && outputs.size() > 0) ||
            outputs.size() == buffer_size) {

            web::json::value json;
            json["targets"] = web::json::value::array(targets);
            json["outputs"] = web::json::value::array(outputs);
            json["label"] = web::json::value::string("train");
            if (epoch < buffer_size) {
                json["rewrite"] = web::json::value::boolean(true);
            }
            targets.clear();
            outputs.clear();

            std::ostringstream request_url;
            request_url << "/update_metrics/" << user_id << "/" << model_id;

            web::http::client::http_client client(U("http://localhost:3000"));
            client.request(web::http::methods::PUT, U(request_url.str()), json);
        }

        // lastTrainNode.gradient = result;
        lastTrainNode.gradient = Blob::ones({{1}});
        lastTrainNode.backward();
        SGD.step();
        lastTrainNode.clear();
    }
}

void predict(int model_id, Graph* graph, std::vector<float>& answer) {
    std::vector<std::vector<float>> predict_data = CsvLoader::load_csv(getPredictPath(model_id));
    graph->ChangeInputData(predict_data[0]);

    // Пока не думаем о нескольких выходах (!) Hard-coded
    auto& lastNode = graph->getLayers(BaseLayerType::PredictOut)[0]->result.value();
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

void extract_from_zip(std::string path, std::string root) {
    zip_t* z;
    int err;
    z = zip_open(path.c_str(), 0, &err);
    if (z == nullptr) {
        throw std::runtime_error("File doesn't exist");
    }
    zip_stat_t info;
    for (int i = 0; i < zip_get_num_files(z); ++i) {
        if (zip_stat_index(z, i, 0, &info) == 0) {
            ofstream fout(root + "/" + info.name, ios::binary);
            zip_file* file = zip_fopen_index(z, i, 0);
            char file_data[info.size];
            zip_fread(file, file_data, info.size);
            fout.write(file_data, info.size);
            fout.close();
        }
    }
    std::filesystem::remove(path);
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

    CROW_ROUTE(app, "/train/<int>/<int>").methods(HTTPMethod::POST)
    ([&](const request& req, int user_id, int model_id) -> response {
        auto body = json::load(req.body);
        std::cout << "Checking json!" << std::endl;
        if (!body) return response(status::BAD_REQUEST, "Invalid body");
        std::cout << "Training" << std::endl;
        if (sessions.find(model_id) != sessions.end() && sessions[model_id] != nullptr) {
            delete sessions[model_id];
        }
        Graph* g = nullptr;

        train(body, &g, user_id, model_id);
        sessions[model_id] = g;
        return response(status::OK, "done");
    });

    // curl -X POST -F "InputFile=@filename" http://0.0.0.0:2000/upload_data/1/0/0
    // Second argument is for request type (train or predict), third - for file type (csv or zip)
    CROW_ROUTE(app, "/upload_data/<int>/<int>/<int>").methods(HTTPMethod::Post)
    ([&](const request& req, int model_id, int type, int file_type) -> response {
        crow::multipart::message file_message(req);
        std::string path, root;
        if (type == 0) {
            path = getDataPath(model_id);
        }
        else {
            path = getPredictPath(model_id);
        }
        root = path;
        if (std::filesystem::exists(path)) {
            std::filesystem::remove_all(path);
        }
        std::filesystem::create_directory(path);
        if (file_type == 0) {
            path += "/1.csv";
        }
        else {
            path += "/1.zip";
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
        if (file_type != 0) {
            try {
                extract_from_zip(path, root);
            }
            catch (...) {
                return response(status::INTERNAL_SERVER_ERROR, "Error in extracting from zip");
            }
        }
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
