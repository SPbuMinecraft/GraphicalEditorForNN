#pragma once

#include <vector>
#include <string>

class CsvLoader {
public:
    static std::vector<std::vector<float>> load_csv(std::string path);
};
