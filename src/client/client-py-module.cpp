#include "client-py-impl.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace dstates::ai;

NB_MODULE(dstates, m) {
    nb::set_leak_warnings(false);
    nb::bind_vector<uint64_list_t>(m, "uint64_list_t");
    nb::bind_vector<string_list_t>(m, "string_list_t");
    nb::module_ ai = m.def_submodule("ai", "AI specific extensions of DataStates");
    nb::bind_vector<tensor_list_t>(ai, "tensor_list_t");
    nb::bind_map<composition_t>(ai, "composition_t");
    nb::class_<py_backend>(ai, "evostore")
      .def(nb::init<const std::string &, const string_list_t &, size_t>())
      .def("save_layers", &py_backend::save_layers)
      .def("load_layers", &py_backend::load_layers)
      .def("store_meta", &py_backend::store_meta)
      .def("get_composition", &py_backend::get_composition)
      .def("get_prefix", &py_backend::get_prefix)
      .def("update_ref_counter", &py_backend::update_ref_counter)
      .def("shutdown", &py_backend::shutdown);
}
