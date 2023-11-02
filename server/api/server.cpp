#include <iostream>
#include <vector>
#include <string>

#include <crow_all.h>

using namespace std;
using namespace crow;


void train(json::rvalue json) {

}

float predict(json::rvalue json) {
    return 0;
}

void invalidArgs() { 
    cout << "Usage: ./server <host: str> <port: int>" << endl;
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc != 3) invalidArgs();
    SimpleApp app;

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

    int port;
    if (sscanf(argv[2], "%d", &port) != 1) invalidArgs();

    // for now we cannot handle even one thread (
    app.port(port).run();

    return 0;
}
