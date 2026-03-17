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

TEST_CASE("Matrix basic operation") {
    Matrix m1;
    Matrix m2(2, 3);
    Matrix m3{ { 1, 2, 3 }, { 4, 5, 6 } };
    size_t m1_rows = 0, m1_cols = 0;
    size_t m2_rows = 2, m2_cols = 3;
    size_t m3_rows = 2, m3_cols = 3;

    SECTION("initialization") {
        REQUIRE(m1.rows() == m1_rows);
        REQUIRE(m1.cols() == m1_cols);
        REQUIRE(m2.rows() == m2_rows);
        REQUIRE(m2.cols() == m2_cols);
        REQUIRE(m3.rows() == m3_rows);
        REQUIRE(m3.cols() == m3_cols);
    }

    SECTION("operator[]") {
        REQUIRE_THROWS_AS(m1[0], std::out_of_range);
        REQUIRE_THROWS_AS(m2[2], std::out_of_range);

        for (size_t i{}; i < m2_rows; ++i)
            for (size_t j{}; j < m2_cols; ++j)
                REQUIRE(m2[i][j] == Catch::Approx(0.0f));

        REQUIRE(m3[0][0] == Catch::Approx(1.0f));
        REQUIRE(m3[0][1] == Catch::Approx(2.0f));
        REQUIRE(m3[0][2] == Catch::Approx(3.0f));
        REQUIRE(m3[1][0] == Catch::Approx(4.0f));
        REQUIRE(m3[1][1] == Catch::Approx(5.0f)); 
        REQUIRE(m3[1][2] == Catch::Approx(6.0f));

        m2[0][1] = 7;
        REQUIRE(m2[0][1] == Catch::Approx(7.0f));
    }

    Matrix m4(m3);
    Matrix m5{ {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15} };
    Matrix m6 = m2;
    Matrix m7{ {1, 2}, {3, 4}, {5, 6}, {7, 8} };
    size_t m4_rows = m3_rows, m4_cols = m3_cols;
    size_t m5_rows = 3, m5_cols = 5;
    size_t m6_rows = m2_rows, m6_cols = m2_cols;
    size_t m7_rows = 4, m7_cols = 2;
    size_t m8_rows = m7_rows, m8_cols = m7_cols;

    SECTION("constructors and assigments") {
        for (size_t i{}; i < m4_rows; ++i)
            for (size_t j{}; j < m4_cols; ++j)
                REQUIRE(m4[i][j] == Catch::Approx(m3[i][j]));

        float val = 1.0f;
        for (size_t i{}; i < m5_rows; ++i)
            for (size_t j{}; j < m5_cols; ++j){
                REQUIRE(m5[i][j] == Catch::Approx(val));
                val += 1.0f;
            }

        for (size_t i{}; i < m6_rows; ++i)
            for (size_t j{}; j < m6_cols; ++j)
                REQUIRE(m6[i][j] == Catch::Approx(m2[i][j]));
            
        val = 1.0f;
        for (size_t i{}; i < m7_rows; ++i)
            for (size_t j{}; j < m7_cols; ++j) {
                REQUIRE(m7[i][j] == Catch::Approx(val));
                val += 1.0f;
            }

        Matrix m8 = std::move(m7);
        val = 1.0f;
        for (size_t i{}; i < m8_rows; ++i)
            for (size_t j{}; j < m8_cols; ++j) {
                REQUIRE(m8[i][j] == Catch::Approx(val));
                val += 1.0f;
            }
        REQUIRE(m7.rows() == 0);
        REQUIRE(m7.cols() == 0);
        REQUIRE(m7.data() == nullptr);
    }
}

TEST_CASE("Matrix arithmetic operation") {
    Matrix m1{ {1, 2, 3}, {4, 5, 6} };
    Matrix m2{ {10, 20, 30},{40, 50, 60} };
    size_t rows = 2, cols = 3;

    SECTION("substrction") {
        Matrix r = m2 - m1;
        REQUIRE(r.rows() == rows);
        REQUIRE(r.cols() == cols);
        for (size_t i{}; i < rows; ++i)
            for (size_t j{}; j < cols; ++j)
                REQUIRE(r[i][j] == Catch::Approx(m2[i][j] - m1[i][j]));

        r -= Matrix{ {1, 1, 1}, {1, 1, 1} };
        REQUIRE(r.rows() == rows);
        REQUIRE(r.cols() == cols);
        for (size_t i{}; i < rows; ++i)
            for (size_t j{}; j < cols; ++j)
                REQUIRE(r[i][j] == Catch::Approx(m2[i][j] - m1[i][j] - 1));
    }

    SECTION("dismesion mismatch") {
        Matrix m3{ {1, 2},{3, 4} };
        REQUIRE_THROWS_AS(m1 - m3, std::invalid_argument);
        REQUIRE_THROWS_AS(m1 -= m3, std::invalid_argument);
    }
}

TEST_CASE("Matrix and Vector") {
    Matrix m{ {1, 2, 3}, {4, 5, 6} };
    Vector v{ 1, 2, 3 };
    size_t m_rows = 2, m_cols = 3;

    SECTION("matrix * vector") {
        Vector r = m * v;
        REQUIRE(r.size() == m_rows);
        REQUIRE(r[0] == Catch::Approx(1*1 + 2*2 + 3*3));
        REQUIRE(r[1] == Catch::Approx(4*1 + 5*2 + 6*3));
    }

    SECTION("vector * matrix") {
        Matrix m2{ {1, 2}, {3, 4}, {5, 6} };
        Vector v2{ 1, 2, 3 };
        Vector r = v2 * m2;
        REQUIRE(r.size() == m2.cols());
        REQUIRE(r[0] == Catch::Approx(1 * 1 + 2 * 3 + 3 * 5));
        REQUIRE(r[1] == Catch::Approx(1 * 2 + 2 * 4 + 3 * 6));
    }

    SECTION("dismesion mismatch") {
        Vector v_wrong{ 1, 2, 3, 4};
        REQUIRE_THROWS_AS(m * v_wrong, std::invalid_argument);
        REQUIRE_THROWS_AS(v_wrong * m, std::invalid_argument);
    }
}

TEST_CASE("Matrix special functions") {
    SECTION("diag") {
        Vector v{ 1, 2, 3 };
        Matrix m = diag(v);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 3);
        for (size_t i{}; i < 3; ++i)
            for (size_t j{}; j < 3; ++j)
                if (i == j)
                    REQUIRE(m[i][j] == Catch::Approx(v[i]));
                else
                    REQUIRE(m[i][j] == Catch::Approx(0.0f));
    }

    SECTION("outer_product") {
        Vector u{ 1, 2, 3 };
        Vector v{ 4, 5 };
        Matrix m = outer_product(u, v);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 2);
        for (size_t i{}; i < 3; ++i)
            for (size_t j{}; j < 2; ++j)
                REQUIRE(m[i][j] == Catch::Approx(u[i] * v[j]));
    }
}