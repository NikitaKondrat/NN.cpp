#pragma once

#include <cstddef>
#include <initializer_list>
#include <functional>

using FtoF = float(*)(float);

class Vector {
private:
    size_t n;
    float* values;
    Vector(size_t, float*);
    void swap(Vector&) noexcept;
    Vector apply_op(const Vector&, const std::function<float(const float&, const float&)>&) const;
public:
    Vector();
    Vector(size_t);
    Vector(std::initializer_list<float>);
    Vector(const Vector&);
    Vector(Vector&&) noexcept;
    ~Vector();
    Vector& operator=(Vector);
    const float& operator[](size_t) const;
    float& operator[](size_t);
    Vector operator+(const Vector&) const;
    Vector& operator+=(const Vector&);
    Vector operator-(const Vector&) const;
    Vector& operator-=(const Vector&);
    Vector& apply(FtoF);
    Vector map(FtoF) const;
    size_t size() const;
    float* data();
    const float* data() const;
};

Vector hadamar(const Vector&, const Vector&);

namespace op {
    float add(const float& x, const float& y);
    float sub(const float& x, const float& y);
}
