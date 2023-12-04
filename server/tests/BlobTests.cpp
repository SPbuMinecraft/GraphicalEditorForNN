#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Allocator.h"
#include "Blob.h"

using namespace std;

/// THESE PASS !!!! ✅✅✅
TEST_CASE("Simple") {
    Allocator::startVirtualMode();
    {
    float p1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float p2[] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
    Blob a = Blob::constBlob(Shape {{2, 5}}, p1);
    Blob b = Blob::constBlob(Shape {{5, 2}}, p2);
    REQUIRE(a.shape.rows() == 2); REQUIRE(a.shape.cols() == 5);
    REQUIRE(b.shape.rows() == 5); REQUIRE(b.shape.cols() == 2);
    
    SUBCASE("Test equals") {
        Blob c = Blob::constBlob(Shape {{2, 5}}, p1);
        CHECK(a == c);
    }
    
    SUBCASE("Test at") {
        CHECK(a(1, 4) == 10);
        a(0, 0) = 5;
        CHECK(a(0, 0) == 5);
    }
    
    SUBCASE("Test transpose") {
        float p3[] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10 };
        Blob c = Blob::constBlob(Shape {{5, 2}}, p3);
        CHECK(a.lazy().transposed() == c);
        CHECK(a.lazy().transposed().transposed() == a);
    }
    
    SUBCASE("Test -") {
        CHECK(-b.lazy() == Blob::constBlob(Shape {{5, 2}}, p1));
        CHECK(-(-b.lazy()) == b);
        float zeros[10] = {0};
        Blob c = Blob::constBlob(Shape {{2, 5}}, zeros);
        Blob some = Blob::constBlob(Shape {{2, 5}}, p2);
        Blob d = -some.lazy();
        CHECK(a.lazy() - d == c);
        Blob more = -(d - a.lazy());
        CHECK(a.lazy() - d == more);
    }
    
    SUBCASE("Test +") {
        float zeros[10] = {0};
        Blob c = Blob::constBlob(Shape {{2, 5}}, zeros);
        Blob d = Blob::constBlob(Shape {{2, 5}}, p2);
        CHECK(a.lazy() + d == c);
        Blob some = d + a.lazy();
        CHECK(a.lazy() + d == some);
        Blob sub = a - d.lazy();
        CHECK(a + -d.lazy() == sub);
    }
    
    SUBCASE("Test stretching +") {
        float fones[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        Blob oneByOne = Blob::constBlob(Shape {{1, 1}}, fones);
        Blob oneByTwo = Blob::constBlob(Shape {{2, 1}}, fones);
        Blob oneByFive = Blob::constBlob(Shape {{1, 5}}, fones);
        Blob ones = Blob::constBlob(Shape {{2, 5}}, fones);
        const LazyBlob& la = a.lazy();
        CHECK(Blob(la + ones) == la + oneByOne);
        CHECK(Blob(la + ones) == la + oneByTwo);
        CHECK(Blob(la + ones) == la + oneByFive);
    }
    
    SUBCASE("Test &") {
        float result[] = { -95, -110, -220, -260 };
        Blob c = Blob::constBlob(Shape {{2, 2}}, result);
        CHECK((a & b.lazy()) == c);
    }
    }

    Allocator::end();
}

TEST_CASE("Medium") {
    // TODO: do
}

TEST_CASE("Hard") {
    // TODO: do
}
