#pragma once

#include <cstddef>
#include <initializer_list>
#include <functional>

using FtoF = std::function<float(float)>;
using FFtoF = std::function<float(float, float)>;

/**
 * Class implementing a coordinate vector and vector operations.
 */
class Vector {
private:
    size_t n;                                                     // number of elements
    float* values;                                                // pointer to element data
    Vector(size_t n_, float* values_);                            // helper constructor from pointer
    void swap(Vector& other) noexcept;                            // for copy-and-swap idiom
    Vector apply_op(const Vector& other, const FFtoF& op) const;  // helper function for binary operations
public:
    /**
     * @defgroup vector_member_funcs Vector Special Member Functions
     * @brief Rule-of-five member functions for resource management.
     * @{
     */
    Vector();
    Vector(size_t);
    Vector(std::initializer_list<float>);
    Vector(const Vector&);
    Vector(Vector&&) noexcept;
    ~Vector();
    Vector& operator=(Vector);
    /** @} */

    /**
     * @defgroup vector_index_ops Vector Index Access Operators
     * @brief Index getter and setter access operators.
     * @param idx Zero-based value index.
     * @return Reference (const or non-const) to the element at the given index.
     * @{
     */
    const float& operator[](size_t idx) const;
    float& operator[](size_t idx);
    /** @} */

    /**
     * @brief Adds two vectors element-wise.
     * @param other Vector to add.
     * @return Resulting vector.
     */
    Vector operator+(const Vector& other) const;

    /**
     * @brief Subtracts two vectors element-wise.
     * @param other Vector to subtract.
     * @return Resulting vector.
     */
    Vector operator-(const Vector& other) const;

    /**
     * @brief Multiplies a vector by a scalar element-wise.
     * @param a Scalar multiplier.
     * @param other Vector to apply multiplication for.
     * @return Resulting vector.
     */
    friend Vector operator*(float a, const Vector& other);

    /**
     * @brief Applies a function to each element of the vector.
     * @param func Transformation function.
     * @return New transformed vector.
     */
    Vector map(const FtoF& func) const;

    /**
     * @brief Vector size getter.
     * @return Number of values in the vector.
     */
    size_t size() const;

    /**
     * @defgroup vector_data_getters Vector Data Accessors
     * @brief Returns pointers to the internal data array.
     * @return Pointer to element data (const or non-const).
     * @{
     */
    float* data();
    const float* data() const;
    /** @} */
};

/**
 * @brief Computes the Hadamard (element-wise) product of two vectors.
 * @param u First vector.
 * @param v Second vector.
 * @return Resulting vector where result[i] = u[i] * v[i].
 */
Vector hadamar(const Vector& u, const Vector& v);

namespace op {
    /**
    * @brief Helper functions used with `apply_op()` vector method.
    */
    float add(float x, float y);
    float sub(float x, float y);
}
