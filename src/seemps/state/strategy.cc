#include <iostream>
#include <vector>
#include "core.h"
#include "tensors.h"
#include "strategy.h"

namespace seemps {

Strategy::Strategy(Truncation a_method, double a_tolerance,
                   Simplification a_simplification_method,
                   double a_simplification_tolerance, size_t a_bond_dimension,
                   int a_num_sweeps, bool a_normalize_flag)
    : method{a_method}, tolerance{a_tolerance},
      simplification_method{a_simplification_method},
      simplification_tolerance{a_simplification_tolerance},
      max_bond_dimension{a_bond_dimension}, max_sweeps{a_num_sweeps},
      normalize{a_normalize_flag} {
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

Strategy Strategy::replace(py::object a_method, py::object a_tolerance,
                           py::object a_simplification_method,
                           py::object a_simplification_tolerance,
                           py::object a_bond_dimension, py::object a_num_sweeps,
                           py::object a_normalize_flag) const {
  Strategy output = *this;
  if (!a_method.is_none()) {
    output.method = py::cast<Truncation>(a_method);
  }
  if (!a_tolerance.is_none()) {
    output.set_tolerance(py::cast<double>(a_tolerance));
  }
  if (!a_simplification_method.is_none()) {
    output.set_simplification_method(
        py::cast<Simplification>(a_simplification_method));
  }
  if (!a_simplification_tolerance.is_none()) {
    output.set_simplification_tolerance(
        py::cast<double>(a_simplification_tolerance));
  }
  if (!a_bond_dimension.is_none()) {
    output.set_max_bond_dimension(py::cast<int>(a_bond_dimension));
  }
  if (!a_num_sweeps.is_none()) {
    output.set_max_sweeps(py::cast<int>(a_num_sweeps));
  }
  if (!a_normalize_flag.is_none()) {
    output.normalize = py::is_true(a_normalize_flag);
  }
  return output;
}

Strategy &Strategy::set_method(Truncation value) {
  method = value;
  return *this;
}

Strategy &Strategy::set_simplification_method(Simplification value) {
  simplification_method = value;
  return *this;
}

Strategy &Strategy::set_tolerance(double value) {
  if (value >= 0 && value <= 1.0) {
    if (value == 0 && method != Truncation::DO_NOT_TRUNCATE) {
      method = Truncation::ABSOLUTE_SINGULAR_VALUE;
    }
    tolerance = value;
    return *this;
  }
  throw std::invalid_argument("Invalid Strategy tolerance");
}

Strategy &Strategy::set_simplification_tolerance(double value) {
  if (value >= 0 && value <= 1.0) {
    simplification_tolerance = value;
    return *this;
  }
  throw std::invalid_argument("Invalid Strategy simplification tolerance");
}

Strategy &Strategy::set_max_bond_dimension(size_t value) {
  max_bond_dimension = value;
  return *this;
}

Strategy &Strategy::set_max_sweeps(int value) {
  if (value > 0) {
    max_sweeps = value;
    return *this;
  }
  throw std::invalid_argument("Invalid Strategy maximum number of sweeps");
}

std::string Strategy::truncation_name() const {
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

std::string Strategy::simplification_name() const {
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

std::string Strategy::str() const {
  std::ostringstream buffer;
  buffer << "Strategy(method=" << truncation_name()
         << ", tolerance=" << tolerance
         << ", max_bond_dimension=" << max_bond_dimension
         << ", normalize=" << (normalize ? "True" : "False")
         << ", simplify=" << simplification_name()
         << ", simplification_tolerance=" << simplification_tolerance
         << ", max_sweeps=" << max_sweeps << ")";
  return buffer.str();
}

static double _truncate_relative_norm_squared(const py::object &a,
                                              const Strategy &s) {
  static std::vector<double> buffer(1024, 0.0);
  size_t N = array_size(a);
  buffer.resize(N + 1);

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
  size_t final_size = 1;
  for (i = 1; i < N; ++i) {
    if (errors[i] > max_error) {
      final_size = N - i + 1;
      break;
    }
  }
  final_size = std::min(final_size, s.get_max_bond_dimension());
  max_error = errors[N - final_size];

#if 0
  if (s.get_normalize_flag()) {
    _normalize(data, final_size);
  }
#endif
  vector_resize_in_place(a, final_size);
  return max_error;
}

static double _truncate_absolute_singular_value(const py::object &a,
                                                const Strategy &s,
                                                double max_error) {
  double *data = array_data<double>(a);
  size_t N = seemps::array_size(a);

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
#if 0
  if (s.get_normalize_flag()) {
    _normalize(data, final_size);
  }
#endif
  vector_resize_in_place(a, final_size);
  return max_error;
}

double destructively_truncate_vector(const py::object a, const Strategy &s) {
  if (!PyArray_Check(a.ptr())) {
    throw std::invalid_argument("truncate_vector expected an ndarray");
  }
  switch (s.get_method()) {
  case Truncation::RELATIVE_NORM_SQUARED_ERROR:
    return _truncate_relative_norm_squared(a, s);
  case Truncation::RELATIVE_SINGULAR_VALUE: {
    return _truncate_absolute_singular_value(
        a, s, s.get_tolerance() * array_data<double>(a)[0]);
  }
  case Truncation::ABSOLUTE_SINGULAR_VALUE:
    return _truncate_absolute_singular_value(a, s, s.get_tolerance());
  case Truncation::DO_NOT_TRUNCATE:
  default:
    return 0.0;
  }
}

} // namespace seemps
