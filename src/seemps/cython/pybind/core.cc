// Here we initialize the MY_PyArray_API
#define INIT_NUMPY_ARRAY_CPP
#include "core.h"
#include "mps.h"
#include "strategy.h"
#include "tensors.h"

using namespace seemps;

const char* __version__ = "pybind11";

static int
ok_loaded()
{
  import_array();
  if (PyErr_Occurred())
    {
      throw std::runtime_error("Failed to import numpy Python module(s)");
    }
  return 1;
}

static void
_cleanup()
{
  numpy_matmul = py::none();
  empty_environment = py::none();
}

PYBIND11_MODULE(pybind, m)
{
  py::object OK_LOADED = py::cast(ok_loaded());

  load_scipy_wrappers();

  py::options options;
  options.disable_function_signatures();

  {
    auto atexit = py::module_::import("atexit");
    auto atexit_register = atexit.attr("register");
    m.def("_cleanup", &_cleanup);
    atexit_register(m.attr("_cleanup"));
  }
  {
    auto numpy = py::module_::import("numpy");
    numpy_matmul = numpy.attr("matmul");
  }
  empty_environment = eye(1);

  m.doc() = "SeeMPS new core routines"; // optional module docstring

  m.def("destructively_truncate_vector", &destructively_truncate_vector,
        "Truncate singular values according to specified criteria, modifying "
        "the array.");

  m.attr("__version__") = __version__;

  py::enum_<Gemm>(m, "GemmOrder")
      .value("NORMAL", GEMM_NORMAL)
      .value("TRANSPOSE", GEMM_TRANSPOSE)
      .value("ADJOINT", GEMM_ADJOINT);
  m.def("_gemm", &gemm);

  m.def("_contract_last_and_first", &contract_last_and_first);

  m.def("_contract_nrjl_ijk_klm", &contract_nrjl_ijk_klm);

  m.def("_destructive_svd", &destructive_svd);

  m.def("_select_svd_driver", &_select_svd_driver);
  m.def("_get_svd_driver", &_get_svd_driver);

  m.attr("STATUS") = OK_LOADED;

  py::class_<Strategy>(m, "Strategy")
      .def(py::init<int, double, int, double, size_t, int, bool>(),
           py::arg("method")
           = static_cast<int>(Truncation::RELATIVE_NORM_SQUARED_ERROR),
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
      .def("get_tolerance", &Strategy::get_tolerance)
      .def("get_simplification_tolerance",
           &Strategy::get_simplification_tolerance)
      .def("get_simplification_method", &Strategy::get_simplification_method)
      .def("get_max_bond_dimension", &Strategy::get_max_bond_dimension)
      .def("get_max_sweeps", &Strategy::get_max_sweeps)
      .def("get_method", &Strategy::get_method)
      .def("get_normalize_flag", &Strategy::get_normalize_flag)
      .def("get_simplify_flag", &Strategy::get_simplify_flag)
      .def("__str__", &Strategy::str);

  py::object DEFAULT_STRATEGY = py::cast(Strategy());
  m.attr("DEFAULT_STRATEGY") = DEFAULT_STRATEGY;

  py::object NO_TRUNCATION
      = py::cast(Strategy()
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
      .value("VARIATIONAL_EXACT_GUESS", Simplification::VARIATIONAL_EXACT_GUESS)
      .export_values();

  m.def("scprod", &scprod,
        R"doc(Compute the scalar product between matrix product states
    :math:`\langle\xi|\psi\rangle`.

    Parameters
    ----------
    bra : MPS
        Matrix-product state for the bra :math:`\xi`
    ket : MPS
        Matrix-product state for the ket :math:`\psi`

    Returns
    -------
    float | complex
        Scalar product.
		 )doc");
  m.def("vdot", &scprod, "Alias for :func:`seemps.state.scprod`");
  m.def(
      "_begin_environment", &_begin_environment, py::arg("D") = int(1),
      R"doc(Initiate the computation of a left environment from two MPS. The bond
    dimension χ defaults to 1. Other values are used for states in canonical
    form that we know how to open and close)doc");
  m.def(
      "_update_left_environment", &_update_left_environment,
      R"doc(Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket.)doc");
  m.def(
      "_update_right_environment", &_update_right_environment,
      R"doc(Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket.)doc");
  m.def("_end_environment", &_end_environment,
        R"doc(Extract the scalar product from the last environment.)doc");
  m.def("_join_environments", &_join_environments,
        R"doc(Join left and right environments to produce a scalar.)doc");

  m.def("_schmidt_weights", &schmidt_weights);
  m.def(
      "_update_in_canonical_form_right", &_update_in_canonical_form_right,
      py::arg("state"), py::arg("tensor"), py::arg("site"), py::arg("strategy"),
      R"doc(Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process)doc");
  m.def(
      "_update_in_canonical_form_left", &_update_in_canonical_form_left,
      py::arg("state"), py::arg("tensor"), py::arg("site"), py::arg("strategy"),
      R"doc(Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process)doc");
  m.def("_canonicalize", &_canonicalize,
        R"doc(Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`.)doc");
  m.def("_recanonicalize", &_recanonicalize,
        R"doc(Update a list of `Tensor3` objects to be in canonical form
    with respect to a new site `center`.)doc");
  m.def("_update_canonical_2site_left", &_update_canonical_2site_left);
  m.def("_update_canonical_2site_right", &_update_canonical_2site_right);
  m.def("_left_orth_2site", &_left_orth_2site);
  m.def("_right_orth_2site", &_right_orth_2site);
}
