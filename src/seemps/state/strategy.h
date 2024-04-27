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
  ~Strategy() = default;

  Strategy(int a_method, double a_tolerance, int a_simplification_method,
           double a_simplification_tolerance, size_t a_bond_dimension,
           int a_num_sweeps, bool a_normalize_flag);

  Strategy replace(py::object a_method, py::object a_tolerance,
                   py::object a_simplification_method,
                   py::object a_simplification_tolerance,
                   py::object a_bond_dimension, py::object a_num_sweeps,
                   py::object a_normalize_flag) const;

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

  Strategy &set_method(int value);
  Strategy &set_simplification_method(int value);
  Strategy &set_tolerance(double value);
  Strategy &set_simplification_tolerance(double value);
  Strategy &set_max_bond_dimension(size_t value);
  Strategy &set_max_sweeps(int value);

  std::string str() const;

private:
  std::string truncation_name() const;
  std::string simplification_name() const;

  static Truncation truncation_from_int(int value);
  static Simplification simplification_from_int(int value);
};

double destructively_truncate_vector(const py::object a, const Strategy &s);

} // namespace seemps
