#include <iostream>
#include <vector>
#include <string>

#include "crow_all.h"

using namespace std;
using namespace crow;


void train(json::rvalue json) {

}

float predict(json::rvalue json) {
    return 0;
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

    CROW_ROUTE(app, "/predict").methods(HTTPMethod::POST)
    ([&](const request& req) -> response {
        auto body = json::load(req.body);
        if (!body) return response(status::BAD_REQUEST, "Invalid body");

        float x;
        try {
            x = predict(body);
        } catch (const std::runtime_error &err) {
            return response(status::BAD_REQUEST, "Invalid body");
        }
        json::wvalue response = {{"answer", x}};
        return crow::response(status::OK, response);
    });

    CROW_ROUTE(app, "/train").methods(crow::HTTPMethod::POST)
    ([](const crow::request& req) -> response {
        auto body = crow::json::load(req.body);
        if (!body) return response(crow::status::BAD_REQUEST, "Invalid body");

        train(body);
        return response(status::OK);
    });

    app.port(2000).multithreaded().run();

    return 0;
}
