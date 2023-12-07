#pragma once

#include <string>
#include "UnshuffledDataLoader.h"
#include "DataLoader.h"

enum class FileExtension {Csv, Png};

class DataMarker {
private:
    UnshuffledDataLoader* train_unshuffled_loader;
    DataLoader train_loader;
    UnshuffledDataLoader* check_unshuffled_loader;
    DataLoader check_loader;
public:
    DataMarker() = default;
    DataMarker(std::string path, FileExtension file_type, int percentage_for_train);
    ~DataMarker();
    DataLoader get_train_loader();
    DataLoader get_check_loader();
};
