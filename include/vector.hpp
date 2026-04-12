#pragma once

#include <cstddef>
#include <initializer_list>
#include <functional>

using FtoF = std::function<float(float)>;
using FFtoF = std::function<float(float, float)>;

class Vector {
private:
    size_t n;
    float* values;
    Vector(size_t, float*);
    void swap(Vector&) noexcept;
    Vector apply_op(const Vector&, const FFtoF&) const;
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
    Vector operator-(const Vector&) const;
    friend Vector operator*(float, const Vector&);
    Vector map(const FtoF&) const;
    size_t size() const;
    float* data();
    const float* data() const;
};

Vector hadamar(const Vector&, const Vector&);

namespace op {
    float add(float x, float y);
    float sub(float x, float y);
}
