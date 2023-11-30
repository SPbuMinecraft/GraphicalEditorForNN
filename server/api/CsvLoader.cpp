#include "CsvLoader.h"
#include <fstream>
#include <boost/algorithm/string.hpp>

std::vector<std::vector<float>> CsvLoader::load_csv(std::string path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("No such csv file in directory");
    }
    std::string line;
    std::vector<std::vector<float>> result;
    getline(fin, line);
    while (!line.empty()) {
        std::vector<std::string> content;
        boost::split(content, line, boost::is_any_of(","));
        result.push_back(std::vector<float>(content.size()));
        for (int i = 0; i < content.size(); ++i) {
            result.back()[i] = std::stof(content[i]);
        }
        getline(fin, line);
    }
    return result;
}