// Here we initialize the MY_PyArray_API
#define INIT_NUMPY_ARRAY_CPP
#include <pybind11/operators.h>
#include "core.h"
#include "tensors.h"
#include "strategy.h"
#include "mps.h"

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

  py::options options;
  options.disable_function_signatures();

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

  m.def("_destructive_svd", &destructive_svd);

  py::object OK_LOADED = py::cast(ok_loaded());

  m.attr("STATUS") = OK_LOADED;

  py::class_<TensorArray3>(m, "TensorArray",
                           R"doc(TensorArray class.

    This class provides the basis for all tensor networks. The class abstracts
    a one-dimensional array of tensors that is freshly copied whenever the
    object is cloned. Two TensorArray's can share the same tensors and be
    destructively modified.

    Parameters
    ----------
    data: Iterable[NDArray]
        Any sequence of tensors that can be stored in this object. They are
        not checked to have the right number of dimensions. This sequence is
        cloned to avoid nasty side effects when destructively modifying it.
    )doc")
      .def(py::init<TensorArray3>())
      .def(py::init<py::list>())
      .def("__getitem__", &TensorArray3::__getitem__)
      .def("__setitem__", &TensorArray3::__setitem__)
      .def("__iter__", &TensorArray3::__iter__)
      .def("__len__", &TensorArray3::len)
      .def_property("_data", &TensorArray3::data, &TensorArray3::set_data)
      .def_property_readonly("size", &TensorArray3::len);

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

  m.attr("__version__") = "c++";

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

  py::class_<MPS, TensorArray3>(m, "MPS",
                                R"doc(MPS (Matrix Product State) class.

    This implements a bare-bones Matrix Product State object with open
    boundary conditions. The tensors have three indices, `A[α,d,β]`, where
    `α,β` are the internal labels and `d` is the physical state of the given
    site.

    Parameters
    ----------
    data : Iterable[Tensor3]
        Sequence of three-legged tensors `A[α,si,β]`. The dimensions are not
        verified for consistency.
    error : float, default=0.0
        Accumulated truncation error in the previous tensors.
										   )doc")
      .def(py::init<const MPS &, double>(), py::arg("state"),
           py::arg("error") = 0.0)
      .def(py::init<py::list, double>(), py::arg("state"),
           py::arg("error") = 0.0)
      .def(
          "copy", &MPS::copy,
          R"doc(Return a shallow copy of the MPS, without duplicating the tensors.)doc")
      .def("as_mps", &MPS::as_mps)
      .def("dimension", &MPS::dimension,
           R"doc(Hilbert space dimension of this quantum system)doc")
      .def("physical_dimensions", &MPS::physical_dimensions,
           R"doc(List of physical dimensions for the quantum subsystems.)doc")
      .def("bond_dimensions", &MPS::bond_dimensions,
           R"doc(List of bond dimensions for the matrix product state.

        Returns a list or vector of `N+1` integers, for an MPS of size `N`.
        The integers `1` to `N-1` are the bond dimensions between the respective
        pairs of tensors. The first and last index are `1`, as it corresponds
        to a matrix product state with open boundary conditions.

        Returns
        -------
        list[int]
            List of virtual bond dimensions between MPS tensors, including the
            boundary ones.

        Examples
        --------
        >>> A = np.ones(1,2,3)
        >>> B = np.ones(3,2,1)
        >>> mps = MPS([A, B])
        >>> mps.bond_dimensions()
        [1, 3, 1])doc")
      .def("norm_squared", &MPS::norm_squared,
           R"doc(Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS.)doc")
      .def("norm", &MPS::norm,
           R"doc(Norm-2 :math:`\\Vert{\\psi}\\Vert^2` of this MPS.)doc")
      .def(
          "zero_state", &MPS::zero_state,
          R"doc(Return a zero wavefunction with the same physical dimensions.)doc")
      .def("left_environment", &MPS::left_environment,
           R"doc(Environment matrix for systems to the left of `site`.)doc")
      .def("right_environment", &MPS::right_environment,
           R"doc(Environment matrix for systems to the right of `site`.)doc")
      .def("error", &MPS::error,
           R"doc(Upper bound of the accumulated truncation error on this state.

        If this quantum state results from `N` steps in which we have obtained
        truncation errors :math:`\\delta_i`, this function returns the estimate
        :math:`\\sqrt{\\sum_{i}\\delta_i^2}`.

        Returns
        -------
        float
			 Upper bound for the actual error when approximating this state.
        )doc")
      .def("set_error", &MPS::set_error)
      .def_property("_error", &MPS::error, &MPS::set_error)
      .def("update_error", &MPS::update_error,
           R"doc(Register an increase in the truncation error.

        Parameters
        ----------
        delta : float
            Error increment in norm-2

        Returns
        -------
        float
            Accumulated upper bound of total truncation error.

        See also
        --------
        :py:meth:`error` : Total accumulated error after this update.
        )doc")
      .def("conj", &MPS::conj,
           "doc(Return the complex-conjugate of this quantum state.)doc")
      .def(
          "__mul__",
          [](const MPS &state, const py::object &weight_or_mps) {
            if (py::isinstance<MPS>(weight_or_mps)) {
              return py::cast(state * weight_or_mps.cast<const MPS &>());
            } else if (py::isinstance<py::int_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::isinstance<py::float_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::iscomplex(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<std::complex<double>>() *
                              state);
            }
            throw py::type_error("Invalid argument to __mul__ by MPS");
          },
          py::is_operator())
      .def(
          "__rmul__",
          [](const MPS &state, const py::object &weight_or_mps) {
            if (py::isinstance<py::int_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::isinstance<py::float_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::iscomplex(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<std::complex<double>>() *
                              state);
            } else if (py::isinstance<MPS>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<const MPS &>() * state);
            }
            throw py::type_error("Invalid argument to __rmul__ by MPS");
          },
          py::is_operator())
      .def(py::self + py::self)
      .def(py::self - py::self)
      // .def(py::self + MPSSum())
      .def(
          "__add__", [](const MPS &a, const MPSSum &b) { return a + b; },
          py::is_operator())
      // .def(py::self - MPSSum())
      .def(
          "__sub__", [](const MPS &a, const MPSSum &b) { return a - b; },
          py::is_operator())
      .def_property_readonly_static("__array_priority__",
                                    [](const py::object &) { return 10000; });

  py::class_<CanonicalMPS, MPS>(m, "CanonicalMPS",
                                R"doc(Canonical MPS class.

    This implements a Matrix Product State object with open boundary
    conditions, that is always on canonical form with respect to a given site.
    The tensors have three indices, `A[α,i,β]`, where `α,β` are the internal
    labels and `i` is the physical state of the given site.

    Parameters
    ----------
    data : Iterable[Tensor3]
        A set of tensors that will be orthogonalized. It can be an
        :class:`MPS` state.
    center : int, optional
        The center for the canonical form. Defaults to the first site
        `center = 0`.
    normalize : bool, optional
        Whether to normalize the state to compensate for truncation errors.
        Defaults to the value set by `strategy`.
    strategy : Strategy, optional
        The truncation strategy for the orthogonalization and later
        algorithms. Defaults to `DEFAULT_STRATEGY`.)doc")
      .def(py::init<const CanonicalMPS &>())
      .def(py::init<const py::list &, py::object, double, py::object,
                    const Strategy &, bool>(),
           py::arg("data"), py::arg("center") = CanonicalMPS::no_defined_center,
           py::arg("error") = 0.0, py::arg("normalize") = py::none(),
           py::arg("strategy") = DEFAULT_STRATEGY,
           py::arg("is_canonical") = false)
      .def(py::init<const MPS &, py::object, double, py::object,
                    const Strategy &, bool>(),
           py::arg("data"), py::arg("center") = CanonicalMPS::no_defined_center,
           py::arg("error") = 0.0, py::arg("normalize") = py::none(),
           py::arg("strategy") = DEFAULT_STRATEGY,
           py::arg("is_canonical") = false)
      .def(py::init<const CanonicalMPS &, py::object, double, py::object,
                    const Strategy &, bool>(),
           py::arg("data"), py::arg("center") = CanonicalMPS::no_defined_center,
           py::arg("error") = 0.0, py::arg("normalize") = py::none(),
           py::arg("strategy") = DEFAULT_STRATEGY,
           py::arg("is_canonical") = false)
      .def_property_readonly("center", &CanonicalMPS::center)
      .def(
          "zero_state", &CanonicalMPS::zero_state,
          R"doc(Return a zero wavefunction with the same physical dimensions.)doc")
      .def(
          "norm_squared", &CanonicalMPS::norm_squared,
          R"doc(Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this CanonicalMPS.)doc")
      .def(
          "norm", &CanonicalMPS::norm,
          R"doc(Norm-2 :math:`\\Vert{\\psi}\\Vert^2` of this CanonicalMPS.)doc")
      .def(
          "left_environment", &CanonicalMPS::left_environment,
          R"doc(Optimized version of :py:meth:`~seemps.state.MPS.left_environment`)doc")
      .def(
          "right_environment", &CanonicalMPS::right_environment,
          R"doc(Optimized version of :py:meth:`~seemps.state.MPS.right_environment)doc")
      .def("Schmidt_weights", &CanonicalMPS::Schmidt_weights,
           py::arg("site") = CanonicalMPS::no_defined_center,
           R"doc(Return the Schmidt weights for a bipartition around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self.center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.

        Returns
        -------
        numbers: np.ndarray
            Vector of non-negative Schmidt weights.)doc")
      .def("entanglement_entropy", &CanonicalMPS::entanglement_entropy,
           py::arg("site") = CanonicalMPS::no_defined_center,
           R"doc(Compute the entanglement entropy of the MPS for a bipartition
        around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self.center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.

        Returns
        -------
        float
            Von Neumann entropy of bipartition.)doc")
      .def("Renyi_entropy", &CanonicalMPS::Renyi_entropy,
           py::arg("site") = CanonicalMPS::no_defined_center,
           py::arg("alpha") = 2.0,
           R"doc(Compute the Renyi entropy of the MPS for a bipartition
        around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self.center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.
        alpha : float, default = 2
            Power of the Renyi entropy.

        Returns
        -------
        float
            Von Neumann entropy of bipartition.)doc")
      .def("update_canonical", &CanonicalMPS::update_canonical)
      .def("update_2site_right", &CanonicalMPS::update_2site_right)
      .def("update_2site_left", &CanonicalMPS::update_2site_left)
      .def("recenter",
           py::overload_cast<int, const Strategy &>(&CanonicalMPS::recenter))
      .def("recenter", py::overload_cast<int>(&CanonicalMPS::recenter))
      .def("normalize_inplace", &CanonicalMPS::normalize_in_place)
      .def("copy", &CanonicalMPS::copy)
      .def(
          "__mul__",
          [](const CanonicalMPS &state, const py::object &weight_or_mps) {
            if (py::isinstance<MPS>(weight_or_mps)) {
              return py::cast(state * weight_or_mps.cast<const MPS &>());
            } else if (py::isinstance<py::int_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::isinstance<py::float_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::iscomplex(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<std::complex<double>>() *
                              state);
            }
            throw py::type_error("Invalid argument to __mul__ by CanonicalMPS");
          },
          py::is_operator())
      .def(
          "__rmul__",
          [](const CanonicalMPS &state, const py::object &weight_or_mps) {
            if (py::isinstance<py::int_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::isinstance<py::float_>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<double>() * state);
            } else if (py::iscomplex(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<std::complex<double>>() *
                              state);
            } else if (py::isinstance<MPS>(weight_or_mps)) {
              return py::cast(weight_or_mps.cast<const MPS &>() * state);
            }
            throw py::type_error(
                "Invalid argument to __rmul__ by CanonicalMPS");
          },
          py::is_operator());

  py::class_<MPSSum>(
      m, "MPSSum",
      R"doc(Class representing a weighted sum (or difference) of two or more :class:`MPS`.

    This class is an intermediate representation for the linear combination of
    MPS quantum states. Assume that :math:`\\psi, \\phi` and :math:`\\xi` are
    MPS and :math:`a, b, c` some real or complex numbers. The addition
    :math:`a \\psi - b \\phi + c \\xi` can be stored as
    `MPSSum([a, -b, c], [ψ, ϕ, ξ])`.


    Parameters
    ----------
    weights : list[Weight]
        Real or complex numbers representing the weights of the linear combination.
    states : list[MPS]
        List of matrix product states weighted.
	   )doc")
      .def(py::init<py::object, py::object, bool>(), py::arg("weights"),
           py::arg("states"), py::arg("check_args") = true)
      .def_property_readonly("weights", &MPSSum::weights)
      .def_property_readonly("states", &MPSSum::states)
      .def_property_readonly("size", &MPSSum::size)
      .def("copy", &MPSSum::copy)
      .def("conj", &MPSSum::conj)
      .def("norm_squared", &MPSSum::norm_squared,
           R"doc(Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS.)doc")
      .def("norm", &MPSSum::norm,
           R"doc(Norm-2 :math:`\\Vert{\\psi}\\Vert^2` of this MPS.)doc")
      .def("error", &MPSSum::error)
      .def("physical_dimensions", &MPSSum::physical_dimensions)
      .def("dimension", &MPSSum::dimension)
      .def(py::self + py::self)
      .def(py::self - py::self)
      // .def(py:self + MPS())
      .def(
          "__add__", [](const MPSSum &a, const MPS &b) { return a + b; },
          py::is_operator())
      // .def(py:self - MPS())
      .def(
          "__sub__", [](const MPSSum &a, const MPS &b) { return a - b; },
          py::is_operator())
      .def(
          "delete_zero_components", &MPSSum::delete_zero_components,
          R"doc(Compute the norm-squared of the linear combination of weights and
    states and eliminate states that are zero or have zero weight.)doc")
      .def_property_readonly_static("__array_priority__",
                                    [](const py::object &) { return 10000; })
      .def("__mul__", &MPSSum::times_object, py::is_operator())
      .def("__rmul__", &MPSSum::times_object, py::is_operator());

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

  m.def("schmidt_weights", &schmidt_weights);
  m.def(
      "_update_in_canonical_form_right", &_update_in_canonical_form_right,
      py::arg("state"), py::arg("tensor"), py::arg("site"), py::arg("strategy"),
      py::arg("overwrite") = false,
      R"doc(Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process)doc");
  m.def(
      "_update_in_canonical_form_left", &_update_in_canonical_form_left,
      py::arg("state"), py::arg("tensor"), py::arg("site"), py::arg("strategy"),
      py::arg("overwrite") = false,
      R"doc(Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process)doc");
  m.def("_canonicalize", &_canonicalize,
        R"doc(Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`.)doc");
  m.def("_update_canonical_2site_left", &_update_canonical_2site_left);
  m.def("_update_canonical_2site_right", &_update_canonical_2site_right);
  m.def("left_orth_2site", &left_orth_2site);

  m.def("right_orth_2site", &right_orth_2site);
}
