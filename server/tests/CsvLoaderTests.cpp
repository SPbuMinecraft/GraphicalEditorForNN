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

TEST_CASE("Csv-load") {
    std::vector<std::vector<float>> and_train = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}};
    std::vector<std::vector<float>> xor_train = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    std::vector<std::vector<float>> predict11 = {{1, 1}};
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
}