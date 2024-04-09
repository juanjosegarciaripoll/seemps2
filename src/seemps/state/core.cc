// Here we initialize the MY_PyArray_API
#define INIT_NUMPY_ARRAY_CPP
#include "core.h"
#include "tensors.h"
#include "strategy.h"

using namespace seemps;

static int ok_loaded() {
  import_array();
  if (PyErr_Occurred()) {
    throw std::runtime_error("Failed to import numpy Python module(s)");
  }
  return 1;
}

PYBIND11_MODULE(core, m) {
  load_scipy_wrappers();

  m.doc() = "SeeMPS new core routines"; // optional module docstring

  m.def("destructively_truncate_vector", &destructively_truncate_vector,
        "Truncate singular values according to specified criteria, modifying "
        "the array.");

  py::enum_<Gemm>(m, "GemmOrder")
      .value("NORMAL", GEMM_NORMAL)
      .value("TRANSPOSE", GEMM_TRANSPOSE)
      .value("ADJOINT", GEMM_ADJOINT);
  m.def("_gemm", &gemm);

  m.def("_contract_last_and_first", &contract_last_and_first);

  m.def("_contract_nrjl_ijk_klm", &contract_nrjl_ijk_klm);

  py::object OK_LOADED = py::cast(ok_loaded());

  m.attr("STATUS") = OK_LOADED;

  py::class_<Strategy>(m, "Strategy")
      .def(py::init<int, double, int, double, size_t, int, bool>(),
           py::arg("method") =
               static_cast<int>(Truncation::RELATIVE_NORM_SQUARED_ERROR),
           py::arg("tolerance") = 1e-8,
           py::arg("simplify") = static_cast<int>(Simplification::VARIATIONAL),
           py::arg("simplification_tolerance") = 1e-8,
           py::arg("max_bond_dimension") = 0x7fffffff,
           py::arg("max_sweeps") = 16, py::arg("normalize") = false)
      .def("replace", &Strategy::replace,
           py::arg("method") = py::cast<py::none>(Py_None),
           py::arg("tolerance") = py::cast<py::none>(Py_None),
           py::arg("simplify") = py::cast<py::none>(Py_None),
           py::arg("simplification_tolerance") = py::cast<py::none>(Py_None),
           py::arg("max_bond_dimension") = py::cast<py::none>(Py_None),
           py::arg("max_sweeps") = py::cast<py::none>(Py_None),
           py::arg("normalize") = py::cast<py::none>(Py_None))
      .def("get_method", &Strategy::get_method)
      .def("get_simplification_method", &Strategy::get_simplification_method)
      .def("get_tolerance", &Strategy::get_tolerance)
      .def("get_simplification_tolerance",
           &Strategy::get_simplification_tolerance)
      .def("get_max_bond_dimension", &Strategy::get_max_bond_dimension)
      .def("get_max_sweeps", &Strategy::get_max_sweeps)
      .def("get_normalize_flag", &Strategy::get_normalize_flag)
      .def("get_simplify_flag", &Strategy::get_simplify_flag)
      .def("__str__", &Strategy::str);

  py::object DEFAULT_STRATEGY = py::cast(Strategy());
  m.attr("DEFAULT_STRATEGY") = DEFAULT_STRATEGY;

  py::object NO_TRUNCATION =
      py::cast(Strategy()
                   .set_method(DO_NOT_TRUNCATE)
                   .set_simplification_method(DO_NOT_SIMPLIFY));
  m.attr("NO_TRUNCATION") = NO_TRUNCATION;

  m.attr("DEFAULT_TOLERANCE") = std::numeric_limits<double>::epsilon();
  m.attr("MAX_BOND_DIMENSION") = 0x7fffffff;

  py::enum_<Truncation>(m, "Truncation")
      .value("DO_NOT_TRUNCATE", Truncation::DO_NOT_TRUNCATE)
      .value("RELATIVE_SINGULAR_VALUE", Truncation::RELATIVE_SINGULAR_VALUE)
      .value("RELATIVE_NORM_SQUARED_ERROR",
             Truncation::RELATIVE_NORM_SQUARED_ERROR)
      .value("ABSOLUTE_SINGULAR_VALUE", Truncation::ABSOLUTE_SINGULAR_VALUE)
      .export_values();

  py::enum_<Simplification>(m, "Simplification")
      .value("DO_NOT_SIMPLIFY", Simplification::DO_NOT_SIMPLIFY)
      .value("CANONICAL_FORM", Simplification::CANONICAL_FORM)
      .value("VARIATIONAL", Simplification::VARIATIONAL)
      .export_values();
}
