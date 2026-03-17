#include <catch2/catch_all.hpp>
#include "vector.hpp"
#include "matrix.hpp"

TEST_CASE("Vector basic operations") {
    Vector v1;
    Vector v2(2);
    Vector v3{0, 1, 2};
    size_t v1_size = 0;
    size_t v2_size = 2;
    size_t v3_size = 3;

    SECTION("initialization") {
        REQUIRE(v1.size() == v1_size);
        REQUIRE(v2.size() == v2_size);
        REQUIRE(v3.size() == v3_size);
    }

    SECTION("operator[]") {
        REQUIRE_THROWS_AS(v1[0], std::out_of_range);
        for (size_t i{}; i < v2_size; ++i)
            REQUIRE(v2[i] == Catch::Approx(0.0f));
        for (size_t i{}; i < v3_size; ++i)
            REQUIRE(v3[i] == Catch::Approx(static_cast<float>(i)));
        v2[1] = 3;
        REQUIRE(v2[1] == Catch::Approx(3.0f));
    }

    Vector v4(v3);
    Vector v5(Vector{0, 1, 2, 3, 4});
    Vector v6 = v2;
    Vector v7 = Vector{0, 1, 2, 3, 4, 5, 6};
    size_t v4_size = v3_size;
    size_t v5_size = 5;
    size_t v6_size = v2_size;
    size_t v7_size = 7;
    size_t v8_size = v7_size;

    SECTION("consturctors & assignments") {
        REQUIRE(v4.size() == v4_size);
        for (size_t i{}; i < v4_size; ++i)
            REQUIRE(v4[i] == Catch::Approx(v3[i]));

        REQUIRE(v5.size() == v5_size);
        for (size_t i{}; i < v5_size; ++i)
            REQUIRE(v5[i] == Catch::Approx(static_cast<float>(i)));

        REQUIRE(v6.size() == v6_size);
        for (size_t i{}; i < v6_size; ++i)
            REQUIRE(v5[i] == Catch::Approx(static_cast<float>(i)));

        REQUIRE(v7.size() == v7_size);
        for (size_t i{}; i < v7_size; ++i)
            REQUIRE(v7[i] == Catch::Approx(static_cast<float>(i)));

        Vector v8 = std::move(v7);
        REQUIRE(v8.size() == v8_size);
        for (size_t i{}; i < v8_size; ++i)
            REQUIRE(v8[i] == Catch::Approx(static_cast<float>(i)));
        REQUIRE(v7.size() == 0);
        REQUIRE(v7.data() == nullptr);

        REQUIRE(v3.size() == v3_size);
        REQUIRE(v2.size() == v2_size);
    }
}

TEST_CASE("Vector arithmetic operations") {
    Vector v{1, 2, 3, 4, 5};
    Vector u{10, 20, 30, 40, 50};
    size_t size = 5;

    SECTION("addition") {
        Vector r = u + v;
        REQUIRE(r.size() == size);
        for (size_t i{1}; i <= size; ++i)
            REQUIRE(r[i - 1] == Catch::Approx(10 * i + i));
        r += Vector{-1, -1, -1, -1, -1};
        REQUIRE(r.size() == size);
        for (size_t i{1}; i <= size; ++i)
            REQUIRE(r[i - 1] == Catch::Approx(10 * i + i - 1));
    }

    SECTION("subtraction") {
        Vector r = u - v;
        REQUIRE(r.size() == size);
        for (size_t i{1}; i <= size; ++i)
            REQUIRE(r[i - 1] == Catch::Approx(10 * i - i));
        r -= Vector{1, 1, 1, 1, 1};
        REQUIRE(r.size() == size);
        for (size_t i{1}; i <= size; ++i)
            REQUIRE(r[i - 1] == Catch::Approx(10 * i - i - 1));

    }
}

TEST_CASE("Vector high order methods") {
    Vector v{1, 2, 3, 4, 5};
    size_t size = 5;
    auto f = [](const float& a) -> float { return 2.0f * a + 1.0f; };

    SECTION("apply") {
        v.apply(f);
        REQUIRE(v.size() == size);
        for (size_t i{1}; i <= size; ++i)
            REQUIRE(v[i - 1] == Catch::Approx(f(i)));
    }

    SECTION("map") {
        Vector r = v.map(f);
        REQUIRE(r.size() == size);
        for (size_t i{1}; i <= size; ++i) {
            REQUIRE(r[i - 1] == Catch::Approx(f(i)));
            REQUIRE(v[i - 1] == Catch::Approx(static_cast<float>(i)));
        }
    }
}