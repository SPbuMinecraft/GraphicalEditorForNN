#pragma once

#include <vector>

class CsvLoader {
public:
    static std::vector<std::vector<float>> load_csv(char* path);
};