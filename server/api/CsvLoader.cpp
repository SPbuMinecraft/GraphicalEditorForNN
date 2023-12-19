#include "CsvLoader.h"
#include <fstream>
#include <sstream>

std::vector<std::vector<float>> CsvLoader::load_csv(std::string path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("No such csv file in directory");
    }
    std::string line;
    std::vector<std::vector<float>> result;
    getline(fin, line);
    while (!line.empty()) {
        std::stringstream line_stream(line);
        result.push_back(std::vector<float>());
        std::string number;
        while (getline(line_stream, number, ',')) {
            result.back().push_back(std::stof(number));
        }
        getline(fin, line);
    }
    return result;
}

std::vector<std::pair<std::string, float>> CsvLoader::load_labels(std::string path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("No such csv file in directory");
    }
    std::string line;
    std::vector<std::pair<std::string, float>> result;
    getline(fin, line);
    while (!line.empty()) {
        std::stringstream line_stream(line);
        std::string file;
        std::string label;
        getline(line_stream, file, ',');
        getline(line_stream, label, ',');
        result.push_back({file, std::stof(label)});
        getline(fin, line);
    }
    return result;
}