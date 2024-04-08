#pragma once
#include "core.h"

namespace seemps {

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

  Strategy(int a_method, double a_tolerance, int a_simplification_method,
           double a_simplification_tolerance, size_t a_bond_dimension,
           int a_num_sweeps, bool a_normalize_flag)
      : method{truncation_from_int(a_method)}, tolerance{a_tolerance},
        simplification_method{simplification_from_int(a_simplification_method)},
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

  Strategy replace(py::object a_method, py::object a_tolerance,
                   py::object a_simplification_method,
                   py::object a_simplification_tolerance,
                   py::object a_bond_dimension, py::object a_num_sweeps,
                   py::object a_normalize_flag) {
    Strategy output = *this;
    if (!a_method.is_none()) {
      output.method = truncation_from_int(py::int_(a_method));
    }
    if (!a_tolerance.is_none()) {
      output.set_tolerance(py::cast<double>(a_tolerance));
    }
    if (!a_simplification_method.is_none()) {
      output.set_simplification_method(
          simplification_from_int(py::int_(a_simplification_method)));
    }
    if (!a_simplification_tolerance.is_none()) {
      output.set_simplification_tolerance(
          py::float_(a_simplification_tolerance));
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

  int get_method() const { return static_cast<int>(method); }

  int get_simplification_method() const {
    return static_cast<int>(simplification_method);
  }

  double get_tolerance() const { return tolerance; }

  double get_simplification_tolerance() const {
    return simplification_tolerance;
  }

  size_t get_max_bond_dimension() const { return max_bond_dimension; }

  bool get_normalize_flag() const { return normalize; }

  int get_max_sweeps() const { return max_sweeps; }

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
    return "Strategy(method=" + truncation_name() +
           ", tolerance=" + std::to_string(tolerance) +
           ", max_bond_dimension=" + std::to_string(max_bond_dimension) +
           ", normalize=" + std::to_string(normalize) +
           ", simplification_method=" + simplification_name() +
           ", max_sweeps=" + std::to_string(max_sweeps) + ")";
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
      throw std::runtime_error(
          "Invalid simplification method found in Strategy");
    }
  }
};

double destructively_truncate_vector(const py::object a, const Strategy &s);

} // namespace seemps
