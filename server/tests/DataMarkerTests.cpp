#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "DataMarker.h"
#include <algorithm>

void check_vectors(std::vector<std::pair<std::vector<float>, float>>& ans, std::vector<std::pair<std::vector<float>, float>>& res) {
    CHECK(ans.size() == res.size());
    for (int i = 0; i < ans.size(); ++i) {
        CHECK(ans[i].first.size() == res[i].first.size());
        CHECK(ans[i].second == res[i].second);
        for (int j = 0; j < ans[i].first.size(); ++j) {
            CHECK(ans[i].first[j] == res[i].first[j]);
        }
    }
}

TEST_CASE("Csv-test") {
    SUBCASE("and-train") {
        DataMarker loader = DataMarker("./tests/data/and-train.csv", FileExtension::Csv, 50);
        DataLoader for_train = loader.get_train_loader();
        DataLoader for_check = loader.get_check_loader();
        std::vector<std::pair<std::vector<float>, float>> ans = {{{0, 0}, 0}, {{0, 1}, 0}, {{1, 0}, 0}, {{1, 1}, 1}};
        std::vector<std::pair<std::vector<float>, float>> res;
        CHECK(for_train.size() == 2);
        CHECK(for_check.size() == 2);
        for (int i = 0; i < 2; ++i) {
            res.push_back(for_train.get_raw(i));
            res.push_back(for_check.get_raw(i));
        }
        sort(res.begin(), res.end());
        check_vectors(ans, res);
    }
    SUBCASE("xor-train") {
        DataMarker loader = DataMarker("./tests/data/xor-train.csv", FileExtension::Csv, 50);
        DataLoader for_train = loader.get_train_loader();
        DataLoader for_check = loader.get_check_loader();
        std::vector<std::pair<std::vector<float>, float>> ans = {{{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 0}};
        std::vector<std::pair<std::vector<float>, float>> res;
        CHECK(for_train.size() == 2);
        CHECK(for_check.size() == 2);
        for (int i = 0; i < 2; ++i) {
            res.push_back(for_train.get_raw(i));
            res.push_back(for_check.get_raw(i));
        }
        sort(res.begin(), res.end());
        check_vectors(ans, res);
    }
}

TEST_CASE("Image-test") {
    DataMarker loader = DataMarker("./tests/data/1", FileExtension::Png, 80);
    DataLoader for_train = loader.get_train_loader();
    DataLoader for_check = loader.get_check_loader();
    std::vector<std::pair<std::vector<float>, float>> ans = {{{255, 255, 255}, 0}, {{0, 0, 0}, 0}, {{159, 252, 253}, 0}, {{255, 255, 0, 0, 255, 255, 0, 0, 0}, 1}, {{0, 255, 100, 153, 136, 255, 0, 174, 100, 217, 0, 255, 0, 201, 100, 234, 21, 255}, 1}};
    std::vector<std::pair<std::vector<float>, float>> res;
    CHECK(for_train.size() == 4);
    CHECK(for_check.size() == 1);
    for (int i = 0; i < 4; ++i) {
        res.push_back(for_train.get_raw(i));
    }
    res.push_back(for_check.get_raw(0));
    sort(res.begin(), res.end());
    sort(ans.begin(), ans.end());
    check_vectors(ans, res);
}