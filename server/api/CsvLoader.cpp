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
