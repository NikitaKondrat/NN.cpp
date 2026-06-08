#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "network.hpp"
#include "vendors.hpp"
#include "utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(bind, m) {
    m.doc() = "Neural Network Bindings";

    m.def("relu", &relu);
    m.def("relu_deriv", &relu_deriv);
    m.def("sigmoid", &sigmoid);
    m.def("sigmoid_deriv", &sigmoid_deriv);
    m.def("bce_lp", &bce_lp, py::arg("est"), py::arg("ans"));

    py::class_<Activation>(m, "Activation")
        .def(py::init<std::function<float(float)>, std::function<float(float)>>());

    py::class_<WeightVendor, std::unique_ptr<WeightVendor, py::nodelete>>(m, "WeightVendor");
    
    py::class_<RandomWeightVendor, WeightVendor>(m, "RandomWeightVendor")
        .def(py::init<size_t, size_t, size_t, size_t, bool, float, float>(),
             py::arg("n_layers"), py::arg("in_size"), py::arg("l_size"),
             py::arg("out_size"), py::arg("with_bias") = false,
             py::arg("a") = -1.0f, py::arg("b") = 1.0f);

    py::class_<ActivationVendor>(m, "ActivationVendor")
        .def(py::init<size_t>(), py::arg("n_layers"))
        .def("set_hid", &ActivationVendor::set_hid, py::return_value_policy::reference_internal)
        .def("set_out", &ActivationVendor::set_out, py::return_value_policy::reference_internal)
        .def("set_l", &ActivationVendor::set_l, py::return_value_policy::reference_internal)
        .def("count", &ActivationVendor::count);

    py::class_<DataVendor, std::unique_ptr<DataVendor, py::nodelete>>(m, "DataVendor")
        .def("count", &DataVendor::count)
        .def("in_size", &DataVendor::in_size)
        .def("out_size", &DataVendor::out_size);

    py::class_<FileDataVendor, DataVendor>(m, "FileDataVendor")
        .def(py::init<const std::string&>(), py::arg("path"));

    py::class_<Network>(m, "Network")
        .def(py::init<const WeightVendor&, const ActivationVendor&, DataVendor*>(),
             py::arg("wv"), py::arg("av"), py::arg("dv"),
             py::keep_alive<1, 3>())
        .def("epochs", &Network::epochs, py::arg("epochs"))
        .def("compute", [](Network& self, const std::vector<float>& input) {
            Vector v(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                v[i] = input[i];
            }
            Vector res = self.compute(v);
            return std::vector<float>(res.data(), res.data() + res.size());
        }, py::arg("input"))
        .def("set_lp", &Network::set_lp, py::arg("loss_fn"), py::return_value_policy::reference_internal)
        .def("set_lr", &Network::set_lr, py::arg("lr"), py::return_value_policy::reference_internal)
        .def("set_wb", &Network::set_wb, py::arg("wb"), py::return_value_policy::reference_internal)
        .def("set_dv", &Network::set_dv, py::arg("dv"), 
             py::return_value_policy::reference_internal, py::keep_alive<1, 2>());
}