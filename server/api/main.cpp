#include "iostream"
#include "vector"
#include "utility"
#include <iostream>

#include "string"
#include "crow.h"
#include "cpp_redis/core/client.hpp"

int main() {
    crow::SimpleApp app;

    CROW_ROUTE(app, "/predict").methods(crow::HTTPMethod::POST)
    ([&](const crow::request& req) -> crow::response {
        auto body = crow::json::load(req.body);
        if (!body)
            return crow::response(400, "Invalid body");
        int x, y;
        try {
            x = std::stoi(body["x"].s());
            y = std::stoi(body["y"].s());
            std::cout << x << " " << y << std::endl;
        } catch (const std::runtime_error &err) {
            return crow::response(400, "Invalid body");
        }
        std::vector<crow::json::wvalue> blogs;
        blogs.push_back(crow::json::wvalue{
                {"answer", x},
                {"loss", y}
        });
        return crow::response(200,  crow::json::wvalue{{"data", blogs}});
    });

    app.port(8080).multithreaded().run();

    return 0;
}
