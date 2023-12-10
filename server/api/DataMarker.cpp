#include <stdexcept>
#include "DataMarker.h"
#include "UnshuffledCsvLoader.h"
#include "UnshuffledImgLoader.h"
#include "Blob.h"

DataMarker::DataMarker(std::string path, FileExtension type, int percentage_for_train, std::size_t batch_size) {
    if (percentage_for_train > 100 || percentage_for_train < 0) {
        throw std::logic_error("Wrong percentage");
    }
    DataLoader file_loader;
    UnshuffledDataLoader* file_unshuffled_loader;
    if (type == FileExtension::Csv) {
        file_unshuffled_loader = new UnshuffledCsvLoader;
        train_unshuffled_loader = new UnshuffledCsvLoader;
        check_unshuffled_loader = new UnshuffledCsvLoader;
    }
    else if (type == FileExtension::Png) {
        file_unshuffled_loader = new UnshuffledImgLoader;
        train_unshuffled_loader = new UnshuffledImgLoader;
        check_unshuffled_loader = new UnshuffledImgLoader;
    }
    else {
        throw std::logic_error("Unsupported type");
    }
    file_loader = DataLoader(file_unshuffled_loader, batch_size, path);
    std::vector<int> rearrangement;
    generate_rearrangement(rearrangement, file_loader.size());
    train_loader = DataLoader(train_unshuffled_loader, batch_size);
    check_loader = DataLoader(check_unshuffled_loader, batch_size);
    int instances_for_train = percentage_for_train * (file_loader.size()) / 100;
    for (int i = 0; i < file_loader.size(); ++i) {
        if (i < instances_for_train) {
            train_loader.add_data(file_loader, rearrangement[i]);
        }
        else {
            check_loader.add_data(file_loader, rearrangement[i]);
        }
    }
    train_loader.shuffle();
    check_loader.shuffle();
    delete file_unshuffled_loader;
}

DataMarker::~DataMarker() {
    delete train_unshuffled_loader;
    delete check_unshuffled_loader;
}

DataLoader DataMarker::get_check_loader() {
    return check_loader;
}

DataLoader DataMarker::get_train_loader() {
    return train_loader;
}