#include "vector.hpp"
#include <stdexcept>
#include <algorithm>

Vector::Vector(size_t n_, float* values_) : n(n_), values(values_ ? values_ : new float[n_]) { }

Vector::Vector() : Vector(0, nullptr) { }

Vector::Vector(size_t n) : Vector(n, new float[n]{}) { }

Vector::Vector(std::initializer_list<float> l) : Vector(l.size(), nullptr) {
    std::copy(l.begin(), l.end(), values);
}

Vector::Vector(const Vector& other) : Vector(other.n, nullptr) {
    std::copy(other.values, other.values + other.n, values);
}

Vector::Vector(Vector&& other) noexcept {
    swap(other);
    other.n = 0;
    other.values = nullptr;
}

Vector::~Vector() {
    delete[] values;
}

Vector& Vector::operator=(Vector other) {
    swap(other);
    return *this;
}

const float& Vector::operator[](size_t idx) const {
    if (idx >= n)
        throw std::out_of_range("vector index out of range");
    return values[idx];
}

float& Vector::operator[](size_t idx) {
    return const_cast<float&>(static_cast<const Vector&>(*this)[idx]);
}

Vector Vector::operator+(const Vector& other) const {
    return apply_op(other, op::add);
}

Vector& Vector::operator+=(const Vector& other) {
    Vector result = apply_op(other, op::add);
    swap(result);
    return *this;
}

Vector Vector::operator-(const Vector& other) const {
    return apply_op(other, op::sub);
}

Vector& Vector::operator-=(const Vector& other) {
    Vector result = apply_op(other, op::sub);
    swap(result);
    return *this;
}

size_t Vector::size() const {
    return n;
}

float* Vector::data() {
    return values;
}

const float* Vector::data() const {
    return values;
}

void Vector::swap(Vector& other) noexcept {
    std::swap(n, other.n);
    std::swap(values, other.values);
}

Vector Vector::apply_op(const Vector& other, const std::function<float(const float&, const float&)>& op) const {
    if (n != other.n) 
        throw std::invalid_argument("same-dimensional vectors required for applied vector operation");
    Vector result(n);
    for (size_t i{}; i < n; ++i)
        result.values[i] = op(values[i], other.values[i]);
    return result;
}

Vector& Vector::apply(const FtoF& func) {
    for (size_t i{}; i < n; ++i)
        values[i] = func(values[i]);
    return *this;
}

Vector Vector::map(const FtoF& func) const {
    Vector result(n);
    for (size_t i{}; i < n; ++i)
        result.values[i] = func(values[i]);
    return result;
}

Vector hadamar(const Vector& u, const Vector& v) {
    if (u.size() != v.size()) 
        throw std::invalid_argument("same-dimensional vectors required for hadamar multiplication");
    Vector result(u.size());
    const float* u_values = u.data();
    const float* v_values = v.data();
    float* result_values = result.data();
    for (size_t i{}; i < u.size(); ++i)
        result_values[i] = u_values[i] * v_values[i];
    return result;
}

float op::add(const float& x, const float& y) {
    return x + y;
}

float op::sub(const float& x, const float& y) {
    return x - y;
}
