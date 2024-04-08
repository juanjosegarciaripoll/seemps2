#include <tuple>
#include <limits>
#include <exception>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/arrayobject.h>

namespace py = pybind11;

enum Truncation {
  DO_NOT_TRUNCATE = 0,
  RELATIVE_SINGULAR_VALUE = 1,
  RELATIVE_NORM_SQUARED_ERROR = 2,
  ABSOLUTE_SINGULAR_VALUE = 3
};

enum Simplification {
  DO_NOT_SIMPLIFY = 0,
  CANONICAL_FORM = 1,
  VARIATIONAL = 2
};

class Strategy {
  Truncation method{Truncation::RELATIVE_NORM_SQUARED_ERROR};
  double tolerance{std::numeric_limits<double>::epsilon()};
  Simplification simplification_method{Simplification::VARIATIONAL};
  double simplification_tolerance{std::numeric_limits<double>::epsilon()};
  size_t max_bond_dimension{0x7fffffff};
  int max_sweeps{16};
  bool normalize{false};

public:

  Strategy() = default;
  Strategy(const Strategy &) = default;
  Strategy(Strategy &&) = default;
  Strategy &operator=(const Strategy &) = default;
  Strategy &operator=(Strategy &&) = default;

  Strategy(int a_method,
		   double a_tolerance,
		   int a_simplification_method,
		   double a_simplification_tolerance,
		   size_t a_bond_dimension,
		   int a_num_sweeps,
		   bool a_normalize_flag):
	method{truncation_from_int(a_method)},
	tolerance{a_tolerance},
	simplification_method{simplification_from_int(a_simplification_method)},
	simplification_tolerance{a_simplification_tolerance},
	max_bond_dimension{a_bond_dimension},
	max_sweeps{a_num_sweeps},
	normalize{a_normalize_flag}
  {
	if (tolerance < 0 || tolerance > 1.0) {
	  throw std::invalid_argument("Invalid Strategy tolerance");
	}
	if (tolerance == 0 && method != Truncation::DO_NOT_TRUNCATE) {
	  method = Truncation::ABSOLUTE_SINGULAR_VALUE;
	}
	if (simplification_tolerance < 0 || simplification_tolerance > 1.0) {
	  throw std::invalid_argument("Invalid Strategy tolerance");
	}
	if (max_sweeps <= 0) {
	  throw std::invalid_argument("Invalid Strategy maximum number of sweeps");
	}
  }

  Strategy
  replace(py::object a_method,
		  py::object a_tolerance,
		  py::object a_simplification_method,
		  py::object a_simplification_tolerance,
		  py::object a_bond_dimension,
		  py::object a_num_sweeps,
		  py::object a_normalize_flag)
  {
	Strategy output = *this;
	if (!a_method.is_none()) {
	  output.method = truncation_from_int(py::int_(a_method));
	}
	if (!a_tolerance.is_none()) {
	  output.set_tolerance(py::cast<double>(a_tolerance));
	}
	if (!a_simplification_method.is_none()) {
	  output.set_simplification_method(simplification_from_int(py::int_(a_simplification_method)));
	}
	if (!a_simplification_tolerance.is_none()) {
	  output.set_simplification_tolerance(py::float_(a_simplification_tolerance));
	}
	if (!a_bond_dimension.is_none()) {
	  output.set_max_bond_dimension(py::int_(a_bond_dimension));
	}
	if (!a_num_sweeps.is_none()) {
	  output.set_max_sweeps(py::int_(a_num_sweeps));
	}
	if (!a_normalize_flag.is_none()) {
	  output.normalize = py::cast<bool>(a_normalize_flag);
	}
	return output;
  }

  int get_method() const {
	return static_cast<int>(method);
  }

  int get_simplification_method() const {
	return static_cast<int>(simplification_method);
  }

  double get_tolerance() const {
	return tolerance;
  }

  double get_simplification_tolerance() const {
	return simplification_tolerance;
  }

  size_t get_max_bond_dimension() const {
	return max_bond_dimension;
  }

  bool get_normalize_flag() const {
	return normalize;
  }

  int get_max_sweeps() const {
	return max_sweeps;
  }

  bool get_simplify_flag() const {
	return simplification_method != Simplification::DO_NOT_SIMPLIFY;
  }

  Strategy &set_method(int value) {
	method = truncation_from_int(value);
	return *this;
  }

  Strategy &set_simplification_method(int value) {
	simplification_method = simplification_from_int(value);
	return *this;
  }

  Strategy &set_tolerance(double value) {
	if (value >= 0 && value <= 1.0) {
	  if (value == 0 && method != Truncation::DO_NOT_TRUNCATE) {
		method = Truncation::ABSOLUTE_SINGULAR_VALUE;
	  }
	  tolerance = value;
	  return *this;
	}
	throw std::invalid_argument("Invalid Strategy tolerance");
  }

  Strategy &set_simplification_tolerance(double value) {
	if (value >= 0 && value <= 1.0) {
	  simplification_tolerance = value;
	  return *this;
	}
	throw std::invalid_argument("Invalid Strategy simplification tolerance");
  }

  Strategy &set_max_bond_dimension(size_t value) {
	max_bond_dimension = value;
	return *this;
  }

  Strategy &set_max_sweeps(int value) {
	if (value > 0) {
	  max_sweeps = value;
	  return *this;
	}
	throw std::invalid_argument("Invalid Strategy maximum number of sweeps");
  }

  Truncation truncation_from_int(int value) {
	if (value < 0 || value > 3) {
	  throw std::invalid_argument("Invalid Strategy Truncation");
	}
	return static_cast<Truncation>(value);
  }

  Simplification simplification_from_int(int value) {
	if (value < 0 || value > 2) {
	  throw std::invalid_argument("Invalid Strategy Simplification");
	}
	return static_cast<Simplification>(value);
  }

  std::string str() const {
	return "Strategy(method=" + truncation_name()
	  + ", tolerance=" + std::to_string(tolerance)
	  + ", max_bond_dimension=" + std::to_string(max_bond_dimension)
	  + ", normalize=" + std::to_string(normalize)
	  + ", simplification_method=" + simplification_name()
	  + ", max_sweeps=" + std::to_string(max_sweeps) + ")";
  }

private:

  std::string truncation_name() const {
	switch (method) {
	case Truncation::DO_NOT_TRUNCATE:
	  return "None";
	case Truncation::RELATIVE_SINGULAR_VALUE:
	  return "RelativeSVD";
	case Truncation::RELATIVE_NORM_SQUARED_ERROR:
	  return "RelativeNorm";
	case Truncation::ABSOLUTE_SINGULAR_VALUE:
	  return "AbsoluteSVD";
	default:
	  throw std::runtime_error("Invalid truncation method found in Strategy");
	}
  }

  std::string simplification_name() const {
	switch (simplification_method) {
	case Simplification::DO_NOT_SIMPLIFY:
	  return "None";
	case Simplification::CANONICAL_FORM:
	  return "CanonicalForm";
	case Simplification::VARIATIONAL:
	  return "Variational";
	default:
	  throw std::runtime_error("Invalid simplification method found in Strategy");
	}
  }
};

static std::tuple<py::array_t<double>, double>
_truncate_do_not_truncate(const py::object &a, const Strategy &s)
{
  return {a, 0.0};
}

static double
_norm(const double *data, size_t size)
{
  double output = 0.0;
  for (; size; --size, ++data) {
	output += (*data)*(*data);
  }
  return std::sqrt(output);
}

static void
_normalize(double *data, size_t size, double norm)
{
  for (; size; --size, ++data) {
	*data /= norm;
  }
}

static void
_normalize(double *data, size_t size)
{
  return _normalize(data, size, _norm(data, size));
}

inline PyArrayObject *
to_array(const py::object &a)
{
  return reinterpret_cast<PyArrayObject*>(a.ptr());
}

inline auto
array_size(const py::object &a)
{
  return static_cast<size_t>(PyArray_Size(a.ptr()));
}

inline auto
array_ndim(const py::object &a)
{
  return PyArray_NDIM(to_array(a));
}

inline auto
array_dim(const py::object &a, int n)
{
  return PyArray_DIM(to_array(a), n);
}

template<class Dimensions>
py::object
array_reshape(const py::object &a, Dimensions &d)
{
  PyArray_Dims dims = {&d[0], static_cast<int>(std::size(d))};
  return py::reinterpret_steal<py::object>(PyArray_Newshape(to_array(a), &dims, NPY_CORDER));
}

auto
as_matrix(const py::object &a, npy_intp rows, npy_intp cols)
{
  npy_intp d[2] = {rows, cols};
  return array_reshape(a, d);
}

auto
matrix_product(const py::object &a, const py::object &b)
{
  return py::reinterpret_steal<py::object>(PyArray_MatrixProduct(a.ptr(), b.ptr()));
}

template<typename elt>
inline elt *
array_data(const py::object &a)
{
  return const_cast<elt *>(static_cast<elt *>(PyArray_DATA(to_array(a))));
}

static std::tuple<py::object, double>
_truncate_relative_norm_squared(const py::object &a, const Strategy &s)
{
  static std::vector<double> buffer(1024, 0.0);
  size_t N = array_size(a);
  buffer.resize(N+1);

  double *errors = &buffer[0];
  double total = 0.0;
  double *data_start = array_data<double>(a);
  double *data = data_start + N;
  size_t i;
  for (i = 1; i <= N; ++i) {
	--data;
	total += data[0] * data[0];
	errors[i] = total;
  }

  double max_error = total * s.get_tolerance();
  double final_error = 0.0;
  size_t final_size = 1;
  for (i = 1; i < N; ++i) {
	if (errors[i] >= max_error) {
	  final_size = N - i + 1;
	  break;
	}
  }
  final_size = std::min(final_size, s.get_max_bond_dimension());
  max_error = errors[N - final_size];

  if (s.get_normalize_flag()) {
	_normalize(data_start, final_size, std::sqrt(total - max_error));
  }
  if (final_size < N) {
	return {a[py::slice(py::int_(0), py::int_(final_size), py::int_(1))], max_error};
  }
  return {a, max_error};
}

static std::tuple<py::object, double>
_truncate_absolute_singular_value(const py::object &a, const Strategy &s, double max_error)
{
  double *data = array_data<double>(a);
  size_t N = array_size(a);
  size_t final_size = N;
  for (size_t i = 0; i < N; ++i) {
	if (data[i] <= max_error) {
	  final_size = i;
	  break;
	}
  }
  if (final_size == 0) {
	final_size = 1;
  } else {
	final_size = std::min(final_size, s.get_max_bond_dimension());
  }
  max_error = 0.0;
  for (size_t i = final_size; i < N; ++i) {
	max_error += data[i] * data[i];
  }
  if (s.get_normalize_flag()) {
	_normalize(data, final_size);
  }
  if (final_size < N) {
	return {a[py::slice(py::int_(0), py::int_(final_size), py::int_(1))], max_error};
  }
  return {a, max_error};
}

static std::tuple<py::object, double>
truncate_vector(const py::object a, const Strategy &s)
{
  if (!PyArray_Check(a.ptr())) {
	throw std::invalid_argument("truncate_vector expected an ndarray");
  }
  switch(s.get_method()) {
  case Truncation::RELATIVE_NORM_SQUARED_ERROR:
	return _truncate_relative_norm_squared(a, s);
  case Truncation::RELATIVE_SINGULAR_VALUE: {
	return _truncate_absolute_singular_value(a, s, s.get_tolerance() * array_data<double>(a)[0]);
  }
  case Truncation::ABSOLUTE_SINGULAR_VALUE:
	return _truncate_absolute_singular_value(a, s, s.get_tolerance());
  case Truncation::DO_NOT_TRUNCATE:
  default:
	return _truncate_do_not_truncate(a, s);
  }
}

static py::object
contract_last_and_first(py::object A, py::object B)
{
  if (!PyArray_Check(A.ptr()) || !PyArray_Check(B.ptr())) {
	throw std::invalid_argument("_contract_last_and_first_expect tensors");
  }

  auto ndimA = array_ndim(A);
  auto Alast = array_dim(A, ndimA - 1);
  auto ndimB = array_ndim(B);
  auto Bfirst = array_dim(B, 0);
  std::vector<npy_intp> new_dims;
  new_dims.reserve(ndimA + ndimB - 2);
  for (int i = 0; i < ndimA - 1; ++i){
	new_dims.emplace_back(array_dim(A, i));
  }
  for (int i = 1; i < ndimB; ++i) {
	new_dims.emplace_back(array_dim(B, i));
  }
  auto Amatrix = as_matrix(A, array_size(A) / Alast, Alast);
  auto Bmatrix = as_matrix(B, Bfirst, array_size(B) / Bfirst);
  auto C = py::reinterpret_steal<py::object>(PyArray_MatrixProduct(Amatrix.ptr(), Bmatrix.ptr()));
  return array_reshape(C, new_dims);
}

py::object numpy, _matmul;

static py::object
contract_nrjl_ijk_klm(py::object U, py::object A, py::object B)
{
  if (PyArray_Check(A.ptr()) == 0 ||
	  PyArray_Check(B.ptr()) == 0 ||
	  PyArray_Check(U.ptr()) == 0 ||
	  array_ndim(A) != 3 ||
	  array_ndim(B) != 3 ||
	  array_ndim(U) != 2) {
	throw std::invalid_argument("Invalid arguments to _contract_nrjl_ijk_klm");
  }
  auto a = array_dim(A, 0);
  auto d = array_dim(A, 1);
  auto b = array_dim(A, 2);
  auto e = array_dim(B, 1);
  auto c = array_dim(B, 2);
  npy_intp final_dims[4] = {a, d, e, c};
  npy_intp intermediate_dims[3] = {a, d*e, c};
  auto AB = matrix_product(as_matrix(A, a*d, b), as_matrix(B, b, e*c));
  return array_reshape(_matmul(U, array_reshape(AB, intermediate_dims)),
					   final_dims);
}

static int
ok_loaded() {
  import_array();
  if (PyErr_Occurred()) {
	throw std::runtime_error("Failed to import numpy Python module(s)");
  }
  numpy = py::module::import("numpy");
  _matmul = numpy.attr("matmul");
  return 1;
}

PYBIND11_MODULE(core, m) {
  m.doc() = "SeeMPS new core routines"; // optional module docstring

  m.def("truncate_vector", &truncate_vector,
		"Truncate singular values according to specified criteria");

  m.def("_contract_last_and_first", &contract_last_and_first);

  m.def("_contract_nrjl_ijk_klm", &contract_nrjl_ijk_klm);

  py::class_<Strategy>(m, "Strategy")
	.def(py::init<int, double, int, double, size_t, int, bool>(),
		 py::arg("method") = static_cast<int>(Truncation::RELATIVE_NORM_SQUARED_ERROR),
		 py::arg("tolerance") = 1e-8,
		 py::arg("simplify") = static_cast<int>(Simplification::VARIATIONAL),
		 py::arg("simplification_tolerance") = 1e-8,
		 py::arg("max_bond_dimension") = 0x7fffffff,
		 py::arg("max_sweeps") = 16,
		 py::arg("normalize") = false)
	.def("replace",
		 &Strategy::replace,
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
	.def("get_simplification_tolerance", &Strategy::get_simplification_tolerance)
	.def("get_max_bond_dimension", &Strategy::get_max_bond_dimension)
	.def("get_max_sweeps", &Strategy::get_max_sweeps)
	.def("get_normalize_flag", &Strategy::get_normalize_flag)
	.def("get_simplify_flag", &Strategy::get_simplify_flag)
	.def("__str__", &Strategy::str);

  py::object OK_LOADED = py::cast(ok_loaded());
  m.attr("STATUS") = OK_LOADED;

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
	.value("RELATIVE_NORM_SQUARED_ERROR", Truncation::RELATIVE_NORM_SQUARED_ERROR)
	.value("ABSOLUTE_SINGULAR_VALUE", Truncation::ABSOLUTE_SINGULAR_VALUE)
	.export_values();

  py::enum_<Simplification>(m, "Simplification")
	.value("DO_NOT_SIMPLIFY", Simplification::DO_NOT_SIMPLIFY)
	.value("CANONICAL_FORM", Simplification::CANONICAL_FORM)
	.value("VARIATIONAL", Simplification::VARIATIONAL)
	.export_values();
}
