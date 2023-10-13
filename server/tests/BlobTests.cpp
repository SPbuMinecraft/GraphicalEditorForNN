#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Blob.h"

using namespace std;

TEST_CASE("Simple") {
    float p1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float p2[] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    Blob a {2, 5, p1};
    Blob b {5, 2, p2};
    REQUIRE(a.rows == 2); REQUIRE(a.cols == 5);
    REQUIRE(b.rows == 5); REQUIRE(b.cols == 2);

    SUBCASE("Test equals") {
        Blob c {2, 5, p1};
        CHECK(a == c);
    }

    SUBCASE("Test copy") {
        Blob c {a};
        CHECK(a == c);
        Blob d = a;
        CHECK(a == d);
        a[0][0] = 10;
        CHECK(a != d);
        CHECK(d == c);
    }

    SUBCASE("Test at") {
        CHECK(a.at(4, 3) == a[4][3]);
        a[0][0] = 5;
        CHECK(a.at(0, 0) == 5);
    }

    SUBCASE("Test transpose") {
        float p3[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10 };
        Blob c {5, 2, p3};
        CHECK(a.transposed() == c);
        CHECK(a.transposed().transposed() == a);
    }

    SUBCASE("Test -") {
        CHECK(-b == Blob {5, 2, p1});
        CHECK(-(-b) == b);
        float zeros[10] = {0};
        Blob c {2, 5, zeros};
        Blob d = -Blob {2, 5, p2};
        CHECK(a - d == c);
        CHECK(a - d == -(d - a));
    }

    SUBCASE("Test +") {
        float zeros[10] = {0};
        Blob c {2, 5, zeros};
        Blob d {2, 5, p2};
        CHECK(a + d == c);
        CHECK(a + d == d + a);
        CHECK(a + -d == a - d);
    }

    SUBCASE("Test stretching +") {
        float fones[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        Blob oneByOne {1, 1, fones};
        Blob oneByTwo {2, 1, fones};
        Blob oneByFive {1, 5, fones};
        Blob ones {2, 5, fones};
        CHECK(a + ones == a + oneByOne);
        CHECK(a + ones == a + oneByTwo);
        CHECK(a + ones == a + oneByFive);
    }

    SUBCASE("Test *") {
        float result[] = { -95, -110, -220, -260 };
        Blob c {2, 2, result};
        CHECK(a * b == c);
    }
}

TEST_CASE("Medium") {
    // TODO: do
}

TEST_CASE("Hard") {
    // TODO: do
}
