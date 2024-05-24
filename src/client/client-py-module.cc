#include "client-py-impl.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;
using namespace nb::literals;
using tensor_list_t = std::vector<nb::ndarray<>>;
using uint64_list_t = std::vector<uint64_t>;
NB_MODULE(modelclient, m) {
  nb::set_leak_warnings(false);
  nb::bind_vector<tensor_list_t>(m, "tensor_list_t");
  nb::bind_vector<uint64_list_t>(m, "uint64_list_t");
  nb::class_<dstatesai::py_backend>(m, "lib_client_backend")
      //.def(nb::init<>())
      .def(nb::init<std::string &, size_t>())
      .def("save_layers", &dstatesai::py_backend::save_layers)
      .def("load_layers", &dstatesai::py_backend::load_layers)
      .def("store_meta", &dstatesai::py_backend::store_meta)
      .def("get_composition", &dstatesai::py_backend::get_composition)
      .def("get_prefix", &dstatesai::py_backend::get_prefix)
      .def("update_ref_counter", &dstatesai::py_backend::update_ref_counter)
      .def("shutdown", &dstatesai::py_backend::shutdown);
}
