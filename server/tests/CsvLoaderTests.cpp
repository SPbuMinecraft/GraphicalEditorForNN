#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CsvLoader.h"

void check_vectors(std::vector<std::vector<float>>& ans, std::vector<std::vector<float>>& res) {
    CHECK(ans.size() == res.size());
    for (int i = 0; i < ans.size(); ++i) {
        CHECK(ans[i].size() == res[i].size());
        for (int j = 0; j < ans[i].size(); ++j) {
            CHECK(ans[i][j] == res[i][j]);
        }
    }
}

void check_labels(std::vector<std::pair<std::string, float>>& ans, std::vector<std::pair<std::string, float>>& res) {
    CHECK(ans.size() == res.size());
    for (int i = 0; i < ans.size(); ++i) {
        CHECK(ans[i].first == res[i].first);
        CHECK(ans[i].second == res[i].second);
    }
}

TEST_CASE("Csv-load") {
    std::vector<std::vector<float>> and_train = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};
    std::vector<std::vector<float>> xor_train = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    std::vector<std::vector<float>> predict11 = {{1, 1}};
    std::vector<std::vector<float>> no_endline = {{1, 1, 1}};
    std::vector<std::vector<float>> big_no_endline = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};
    SUBCASE("and_train") {
        std::vector<std::vector<float>> result = CsvLoader::load_csv("./tests/data/and-train.csv");
        check_vectors(and_train, result);
    }
    SUBCASE("xor_train") {
        std::vector<std::vector<float>> result = CsvLoader::load_csv("./tests/data/xor-train.csv");
        check_vectors(xor_train, result);
    }
    SUBCASE("predict") {
        std::vector<std::vector<float>> result = CsvLoader::load_csv("./tests/data/predict11.csv");
        check_vectors(predict11, result);
    }
    SUBCASE("No_endline") {
        std::vector<std::vector<float>> result = CsvLoader::load_csv("./tests/data/no_endline.csv");
        check_vectors(no_endline, result);
    }
    SUBCASE("Big_no_endline") {
        std::vector<std::vector<float>> result = CsvLoader::load_csv("./tests/data/big_no_endline.csv");
        check_vectors(big_no_endline, result);
    }
}

TEST_CASE("Load-labels") {
    std::vector<std::pair<std::string, float>> endline_ans = {{"zero", 0}, {"one", 1}};
    std::vector<std::pair<std::string, float>> no_endl_ans = {{"zero", 0}, {"one", 1}};
    SUBCASE("Endline") {
        std::vector<std::pair<std::string, float>> result = CsvLoader::load_labels("./tests/data/labels.csv");
        check_labels(endline_ans, result);
    }
    SUBCASE("No_endline") {
        std::vector<std::pair<std::string, float>> result = CsvLoader::load_labels("./tests/data/labels_no_endl.csv");
        check_labels(no_endl_ans, result);
    }
}
