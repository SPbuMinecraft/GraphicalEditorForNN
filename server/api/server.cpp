#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>

#include <zip.h>
#include <filesystem>
#include <crow_all.h>

#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
#include <cpprest/uri.h>
#include <cpprest/json.h>

#pragma push_macro("U")
#undef U
#include "ImageLoader.h"
#pragma pop_macro("U")

#include "GraphBuilder.h"
#include "CsvLoader.h"
#include "DataMarker.h"

using namespace std;
using namespace crow;

std::string getDataPath(int id) {
    return "./model_data/data/" + std::to_string(id);
}

std::string getPredictPath(int id) {
    return "./model_data/predict/" + std::to_string(id);
}

void GetLogs(const Blob& node, std::vector<web::json::value>& values) {
    values.reserve(values.size() + node.shape.size());
    for (size_t sample_index = 0; sample_index < node.shape.dim4(); ++sample_index) {
        for (size_t feature_index = 0; feature_index < node.shape.cols(); ++feature_index) {
            values.push_back(web::json::value::number(node(sample_index, 0, 0, feature_index)));
        }
    }
}

void train(json::rvalue& json, Graph** graph, int model_id, int user_id, FileExtension extension) {
    Allocator::end();

    RandomObject initObject(0, 1, 42);
    OptimizerBase SGD = OptimizerBase(0.1);

    // Should be adopted for DataLoader possibilities
    std::string path = getDataPath(model_id);
//    if (extension == FileExtension::Csv) {
//        path += "/1.csv";
//    }

    std::cerr << "start data processing" << std::endl;
    DataMarker dataMarker = DataMarker(path, extension, 100, 4);
    DataLoader dataLoader = dataMarker.get_train_loader();
    std::cerr << "stop data processing" << std::endl;

    size_t batch_size = 4;  // hard-coded
    *graph = new Graph();
    (*graph)->Initialize(json, &initObject, SGD, batch_size);
    std::cout << "Graph is ready!" << std::endl;

    auto& lastTrainNode = (*graph)->getLayers(BaseLayerType::TrainOut)[0]->result.value();

    lastTrainNode.forward();
    lastTrainNode.gradient = Blob::ones({{1}});
    lastTrainNode.backward();
    Allocator::endSession();
    lastTrainNode.clear();
    Allocator::endVirtualMode();

    size_t buffer_size = 2, actual_size = 0;
    web::json::value request;
    request[U("rewrite")] = web::json::value::boolean(true);
    std::vector<web::json::value> targets, outputs;

    size_t max_epochs = 5;
    std::pair<std::vector<float>, std::vector<float>> batch;

    web::http::client::http_client client(U("http://localhost:3000"));
    std::ostringstream request_url;
    request_url << "/update_metrics/" << user_id << "/" << model_id;

    auto& lastPredictNode = (*graph)->getLayers(BaseLayerType::PredictOut)[0]->result.value().output;
    auto& targetsNode = (*graph)->getLayers(BaseLayerType::Targets)[0]->result.value().output;
    for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
        std::cerr << epoch << std::endl;
        for (size_t batch_index = 0; batch_index < dataLoader.batch_count(); ++batch_index) {
            std::cerr << "\t" << batch_index << std::endl;
            batch = dataLoader.get_raw(batch_index);
            (*graph)->ChangeLayersData(batch.first, BaseLayerType::Data);
            (*graph)->ChangeLayersData(batch.second, BaseLayerType::Targets);

            lastTrainNode.forward();

            GetLogs(lastPredictNode.value(), outputs);
            GetLogs(targetsNode.value(), targets);

            lastTrainNode.gradient = Blob::ones({{1}});
            lastTrainNode.backward();
            SGD.step();
            Allocator::endSession();
            lastTrainNode.clear();
        }

        request[U("targets")][actual_size] = web::json::value::array(targets);
        request[U("outputs")][actual_size] = web::json::value::array(outputs);
        targets.clear();
        outputs.clear();
        ++actual_size;

        if ((epoch == max_epochs - 1 && actual_size > 0) ||
            actual_size == buffer_size) {
            request[U("label")] = web::json::value::string("train");
            client.request(web::http::methods::PUT, U(request_url.str()), request);
            request = web::json::value();
            actual_size = 0;
        }
    }
}

void predict(int model_id, Graph* graph, std::vector<float>& answer, FileExtension extension) {
    std::vector<std::vector<float>> predict_data;
    if (extension == FileExtension::Csv) {
        predict_data = CsvLoader::load_csv(getPredictPath(model_id) + "/1.csv");
    }
    else {
        predict_data = {ImageLoader::load_image((getPredictPath(model_id) + "/1.png").c_str())};
    }
    graph->ChangeLayersData(predict_data[0], BaseLayerType::Data);

    // Пока не думаем о нескольких выходах (!) Hard-coded
    auto& lastNode = graph->getLayers(BaseLayerType::PredictOut)[0]->result.value();
    lastNode.clear();
    lastNode.forward();

    auto& result = lastNode.output.value();

    answer.reserve(result.shape.cols());
    for (size_t i = 0; i < result.shape.cols(); ++i) {
        answer.push_back(result(0, 0, 0, i));
        std::cout << result(0, 0, 0, i) << std::endl;
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
    for (int i = 0; i < zip_get_num_entries(z, ZIP_FL_UNCHANGED); ++i) {
        if (zip_stat_index(z, i, 0, &info) == 0) {
            ofstream fout(root + "/" + info.name, ios::binary);
            zip_file* file = zip_fopen_index(z, i, 0);
            char *file_data = new char[info.size];
            zip_fread(file, file_data, info.size);
            fout.write(file_data, info.size);
            delete[] file_data;
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
    std::map<int, FileExtension> file_types;

    CROW_ROUTE(app, "/predict/<int>").methods(HTTPMethod::GET)
    ([&](const request& req, int model_id) -> response {
        (void)req;
        if (sessions.find(model_id) == sessions.end()) return response(status::METHOD_NOT_ALLOWED, "Not trained");
        std::vector<float> answer;
        try {
            predict(model_id, sessions[model_id], answer, file_types[model_id]);
        } catch (const std::runtime_error &err) {
            std::cout << err.what() << std::endl;
            return response(status::BAD_REQUEST, "Invalid body");
        }
        json::wvalue response = (int)(answer[1] > answer[0]);
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

        train(body, &g, model_id, user_id, file_types[model_id]);
        sessions[model_id] = g;
        return response(status::OK, "done");
    });

    // curl -X POST -F "InputFile=@filename" http://0.0.0.0:2000/upload_data/1/0
    // Second argument is for request type (train or predict), third - for file type (csv or zip)
    CROW_ROUTE(app, "/upload_data/<int>/<int>").methods(HTTPMethod::Post)
    ([&](const request& req, int model_id, int type) -> response {
        string content_type = req.get_header_value("Content-Type");
        if (content_type.empty()) return response(status::BAD_REQUEST, "No Content-Type header provided");

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

        std::cerr << content_type << std::endl;
        if (content_type == "text/csv") {
            path += "/1.csv";
            file_types[model_id] = FileExtension::Csv;
        }
        else if (content_type == "application/zip") {
            path += "/1.zip";
            file_types[model_id] = FileExtension::Png;
        } else if (content_type == "image/png") {
            path += "/1.png";
            file_types[model_id] = FileExtension::Png;
        } else {
            return response(status::BAD_REQUEST, "Invalid content type: " + content_type);
        }

        std::ofstream out_file(path, ios::binary);
        out_file << req.body;
        out_file.close();

        if (content_type == "application/zip" && type == 0) {
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
