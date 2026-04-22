#include <iostream>
#include <string>
#include <iomanip>
#include <cmath>
#include "network.hpp"

void train_bool_op(const std::string& s, DataVendor* dv) {
    size_t n_layers = 4;
    size_t in_size = 2;
    size_t l_size = 3;
    size_t out_size = 1;
    bool with_bias = false;
    size_t epochs = 100'000;

    RandomWeightVendor wv(n_layers, in_size, l_size, out_size, with_bias);

    ActivationVendor av(n_layers);
    Activation activation(sigmoid, sigmoid_deriv);
    av.set_hid(activation).set_out(activation);

    Network nw(wv, av, dv);

    nw.epochs(epochs);

    Vector est(dv->count());
    for (size_t i{}; i < dv->count(); ++i)
        est[i] = nw.compute(dv->fetch(i).first)[0];

    for (int i{0}; i <= 1; ++i)
        for (int j{0}; j <= 1; ++j)
            std::cout << i << " " << s << " " << j << " = " << est[2 * i + j] << std::endl;
    
}

void train_binary_clsf_line(DataVendor* dv) {
    size_t n_layers = 4;
    size_t in_size = 2;
    size_t l_size = 16;
    size_t out_size = 1;
    size_t epochs = 25'000;
    bool with_bias = true;

    RandomWeightVendor wv(n_layers, in_size, l_size, out_size, with_bias);

    ActivationVendor av(n_layers);
    av.set_hid(Activation(relu, relu_deriv)).set_out(Activation(sigmoid, sigmoid_deriv));

    Network nw(wv, av, dv);
    nw.set_wb(with_bias).set_lp(bce_lp).set_lr(0.01f);

    nw.epochs(epochs);

    std::cout << std::fixed << std::setprecision(6) << std::right;
    for (int i{-4}; i < 20; ++i) {
        for (int j{-4}; j < 20; ++j) {
            float est = nw.compute({static_cast<float>(i), static_cast<float>(j)})[0];
            float ans = (i + j - 7 >= 0);
            if (std::abs(est - ans) > 0.3f)
                std::cout << "warning! ";
            std::cout << "(" << std::setw(4) << i << ", " << std::setw(4) << j << ")"
                      << " est: " << std::setw(10) << est 
                      << " | ans: " << std::setw(10) << ans << std::endl;
        }
    }
}

void train_binary_clsf_semicircle(DataVendor* dv) {
    size_t n_layers = 4;
    size_t in_size = 2;
    size_t l_size = 16;
    size_t out_size = 1;
    size_t epochs = 25'000;
    bool with_bias = true;

    RandomWeightVendor wv(n_layers, in_size, l_size, out_size, with_bias);

    ActivationVendor av(n_layers);
    av.set_hid(Activation(relu, relu_deriv)).set_out(Activation(sigmoid, sigmoid_deriv));

    Network nw(wv, av, dv);
    nw.set_wb(with_bias).set_lp(bce_lp).set_lr(0.01f);

    nw.epochs(epochs);

    std::cout << std::fixed << std::setprecision(6) << std::right;
    for (float i{-1.25f}; i <= 1.25f; i += 0.25f) {
        for (float j{.0f}; j <= 1.25f; j += 0.25f) {
            float est = nw.compute({static_cast<float>(i), static_cast<float>(j)})[0];
            float ans = (i * i + j * j >= 1);
            if (std::abs(est - ans) > 0.3f)
                std::cout << "warning! ";
            std::cout << "(" << std::setw(4) << i << ", " << std::setw(4) << j << ")"
                      << " est: " << std::setw(10) << est 
                      << " | ans: " << std::setw(10) << ans << std::endl;
        }
    }
}

int main() {
    ObjectDataVendor xor_ds = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}},
    };
    ObjectDataVendor and_ds = {
        {{0, 0}, {0}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{1, 1}, {1}},
    };
    ObjectDataVendor or_ds = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {1}},
    };

    std::cout << "========== Binary operations training ==========" << std::endl;
    train_bool_op("^", &xor_ds);
    std::cout << std::endl;
    train_bool_op("&", &and_ds);
    std::cout << std::endl;
    train_bool_op("|", &or_ds);
    std::cout << std::endl;

    ObjectDataVendor linear_ds = {
        // Класс 0: точки ниже прямой x + y = 7
        {{0, 0}, {0}},
        {{0, 1}, {0}},
        {{1, 0}, {0}},
        {{0, 2}, {0}},
        {{2, 0}, {0}},
        {{1, 1}, {0}},
        {{0, 3}, {0}},
        {{3, 0}, {0}},
        {{1, 2}, {0}},
        {{2, 1}, {0}},
        
        // Класс 1: точки выше или на прямой x + y = 7
        {{7, 0}, {1}},
        {{0, 7}, {1}},
        {{8, 0}, {1}},
        {{0, 8}, {1}},
        {{4, 4}, {1}},
        {{9, 0}, {1}},
        {{0, 9}, {1}},
        {{5, 3}, {1}},
        {{3, 5}, {1}},
        {{6, 2}, {1}},
    };

    std::cout << "========== Binary classification training (line) ==========" << std::endl;
    train_binary_clsf_line(&linear_ds);
    std::cout << std::endl;

    ObjectDataVendor semicircle_ds = {
        // Класс 0: внутри полукруга (x² + y² < 1)
        {{0.0f, 0.0f}, {0}},
        {{0.3f, 0.0f}, {0}},
        {{-0.3f, 0.0f}, {0}},
        {{0.0f, 0.3f}, {0}},
        {{0.2f, 0.2f}, {0}},
        {{-0.2f, 0.2f}, {0}},
        {{0.4f, 0.3f}, {0}},
        {{-0.4f, 0.3f}, {0}},
        {{0.3f, 0.4f}, {0}},
        {{-0.3f, 0.4f}, {0}},
        {{0.5f, 0.4f}, {0}},
        {{-0.5f, 0.4f}, {0}},
        {{0.6f, 0.3f}, {0}},
        {{-0.6f, 0.3f}, {0}},
        {{0.0f, 0.6f}, {0}},
        {{0.1f, 0.5f}, {0}},
        {{-0.1f, 0.5f}, {0}},

        // Класс 1: снаружи полукруга (x² + y² > 1)
        {{1.5f, 0.0f}, {1}},
        {{-1.5f, 0.0f}, {1}},
        {{0.0f, 1.5f}, {1}},
        {{1.2f, 1.2f}, {1}},
        {{-1.2f, 1.2f}, {1}},
        {{1.3f, 0.5f}, {1}},
        {{-1.3f, 0.5f}, {1}},
        {{0.5f, 1.3f}, {1}},
        {{-0.5f, 1.3f}, {1}},
        {{1.0f, 1.0f}, {1}},
        {{-1.0f, 1.0f}, {1}},
        {{0.8f, 1.2f}, {1}},
        {{-0.8f, 1.2f}, {1}},
    };

    std::cout << "========== Binary classification training (semicircle) ==========" << std::endl;
    train_binary_clsf_semicircle(&semicircle_ds);
    std::cout << std::endl;
}