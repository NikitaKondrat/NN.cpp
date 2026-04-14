#pragma once

#include <cstddef>
#include <initializer_list>
#include <functional>
#include "vector.hpp"

using Row = Vector;

/**
 * @brief Class implementing a matrix and matrix operations.
 * 
 * The matrix is stored as an array of `Row` (Vector) objects.
 */
class Matrix {
private:
    size_t n_rows, n_cols;                                        // numbers of rows and columns
    Row* matrix;                                                  // pointer to array of row vectors
    Matrix(size_t n_rows_, size_t n_cols_, Row* matrix_);         // helper constructor from pointer
    void swap(Matrix& other) noexcept;                            // for copy-and-swap idiom
    Matrix apply_op(const Matrix& other, const FFtoF& op) const;  // helper function for binary operations
public:
    /**
     * @defgroup matrix_member_funcs Matrix Special Member Functions
     * @brief Rule-of-five member functions for resource management.
     * @{
     */
    Matrix();
    Matrix(size_t n_rows, size_t n_cols);
    Matrix(std::initializer_list<Row> l);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    ~Matrix();
    Matrix& operator=(Matrix other);
    /** @} */

    /**
     * @defgroup matrix_index_ops Matrix Index Access Operators
     * @brief Index getter and setter access operators.
     * @param idx Zero-based value index.
     * @return Reference (const or non-const) to the row vector at the given index.
     * @{
     */
    const Row& operator[](size_t idx) const;
    Row& operator[](size_t idx);
    /** @} */

    /**
     * @brief Subtracts two matrices element-wise.
     * @param other Matrix to subtract.
     * @return Resulting matrix.
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Multiplies matrix by a column vector.
     * @param v Column vector (size must match `cols()`).
     * @return Resulting vector of size `rows()`.
     */
    Vector operator*(const Vector& v) const;

    /**
     * @brief Multiplies matrix by a scalar element-wise.
     * @param a Scalar multiplier.
     * @param matrix Matrix to multiply.
     * @return Resulting matrix.
     */
    friend Matrix operator*(float a, const Matrix& matrix);

    /**
     * @brief Applies a function to each element of the matrix.
     * @param func Transformation function.
     * @return New transformed matrix.
     */
    Matrix map(const FtoF& func) const;

    /**
     * @brief Number of rows getter.
     * @return Number of rows.
     */
    size_t rows() const;

    /**
     * @brief Number of columns getter.
     * @return Number of columns.
     */
    size_t cols() const;

    /**
     * @defgroup matrix_data_getters Matrix Data Accessors
     * @brief Returns pointers to the internal row array.
     * @return Pointer to row data (const or non-const).
     * @{
     */
    Row* data();
    const Row* data() const;
    /** @} */
};

/**
 * @brief Creates a diagonal matrix from a vector.
 * @param v Vector whose elements become the diagonal.
 * @return Square matrix with `v[i]` at position `(i, i)` and zeros elsewhere.
 */
Matrix diag(const Vector& v);

/**
 * @brief Computes the outer product of two vectors.
 * @param u Column vector (size m).
 * @param v Row vector (size n).
 * @return Matrix of size m×n where result[i][j] = u[i] * v[j].
 */
Matrix outer_product(const Vector& u, const Vector& v);

/**
 * @brief Multiplies a row vector by a matrix.
 * @param v Row vector (size must match `matrix.rows()`).
 * @param matrix Matrix to multiply.
 * @return Resulting vector of size `matrix.cols()`.
 */
Vector operator*(const Vector& v, const Matrix& matrix);