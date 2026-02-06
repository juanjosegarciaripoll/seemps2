#pragma once
#include "core.h"

namespace seemps
{

enum Truncation
{
  DO_NOT_TRUNCATE = 0,
  RELATIVE_SINGULAR_VALUE = 1,
  RELATIVE_NORM_SQUARED_ERROR = 2,
  ABSOLUTE_SINGULAR_VALUE = 3
};

enum Simplification
{
  DO_NOT_SIMPLIFY = 0,
  CANONICAL_FORM = 1,
  VARIATIONAL = 2,
  VARIATIONAL_EXACT_GUESS = 3
};

inline constexpr double DEFAULT_TOLERANCE
    = std::numeric_limits<double>::epsilon();
inline constexpr size_t MAX_BOND_DIMENSION = 0x7fffffff;

class Strategy
{
public:
  // Function pointer typedef for truncation methods
  using TruncationFunction = double (*)(const py::object&, const Strategy&);

private:
  // Static truncation method implementations
  static double truncate_do_not_truncate(const py::object& a,
                                         const Strategy& s);
  static double truncate_relative_norm_squared(const py::object& a,
                                               const Strategy& s);
  static double truncate_relative_singular_value(const py::object& a,
                                                 const Strategy& s);
  static double truncate_absolute_singular_value(const py::object& a,
                                                 const Strategy& s);

  // Helper to select the appropriate truncation function based on method
  TruncationFunction select_truncation_function() const;

  Truncation method{ Truncation::RELATIVE_NORM_SQUARED_ERROR };
  double tolerance{ DEFAULT_TOLERANCE };
  Simplification simplification_method{ Simplification::VARIATIONAL };
  double simplification_tolerance{ DEFAULT_TOLERANCE };
  size_t max_bond_dimension{ MAX_BOND_DIMENSION };
  int max_sweeps{ 16 };
  bool normalize{ false };
  TruncationFunction truncation_function{ truncate_do_not_truncate };

  static Truncation truncation_from_int(int value);
  static Simplification simplification_from_int(int value);

public:
  Strategy() : truncation_function(select_truncation_function()) {}
  Strategy(const Strategy&) = default;
  Strategy(Strategy&&) = default;
  Strategy& operator=(const Strategy&) = default;
  Strategy& operator=(Strategy&&) = default;

  Strategy(int a_method, double a_tolerance, int a_simplification_method,
           double a_simplification_tolerance, size_t a_bond_dimension,
           int a_num_sweeps, bool a_normalize_flag)
      : method{ truncation_from_int(a_method) }, tolerance{ a_tolerance },
        simplification_method{ simplification_from_int(
            a_simplification_method) },
        simplification_tolerance{ a_simplification_tolerance },
        max_bond_dimension{ a_bond_dimension }, max_sweeps{ a_num_sweeps },
        normalize{ a_normalize_flag },
        truncation_function{ select_truncation_function() }
  {
    if (tolerance < 0 || tolerance > 1.0)
      {
        throw std::invalid_argument("Invalid Strategy tolerance");
      }
    if (tolerance == 0 && method != Truncation::DO_NOT_TRUNCATE)
      {
        method = Truncation::ABSOLUTE_SINGULAR_VALUE;
      }
    if (simplification_tolerance < 0 || simplification_tolerance > 1.0)
      {
        throw std::invalid_argument("Invalid Strategy tolerance");
      }
    if (max_sweeps <= 0)
      {
        throw std::invalid_argument(
            "Invalid Strategy maximum number of sweeps");
      }
  }

  Strategy replace(py::object a_method, py::object a_tolerance,
                   py::object a_simplification_method,
                   py::object a_simplification_tolerance,
                   py::object a_bond_dimension, py::object a_num_sweeps,
                   py::object a_normalize_flag) const;

  int
  get_method() const
  {
    return static_cast<int>(method);
  }

  int
  get_simplification_method() const
  {
    return static_cast<int>(simplification_method);
  }

  double
  get_tolerance() const
  {
    return tolerance;
  }

  double
  get_simplification_tolerance() const
  {
    return simplification_tolerance;
  }

  size_t
  get_max_bond_dimension() const
  {
    return max_bond_dimension;
  }

  bool
  get_normalize_flag() const
  {
    return normalize;
  }

  int
  get_max_sweeps() const
  {
    return max_sweeps;
  }

  bool
  get_simplify_flag() const
  {
    return simplification_method != Simplification::DO_NOT_SIMPLIFY;
  }

  // Execute the truncation using the stored function pointer
  double
  truncate(const py::object& a) const
  {
    return truncation_function(a, *this);
  }

  Strategy&
  set_method(int value)
  {
    method = truncation_from_int(value);
    truncation_function = select_truncation_function();
    return *this;
  }

  Strategy&
  set_simplification_method(int value)
  {
    simplification_method = simplification_from_int(value);
    return *this;
  }

  Strategy&
  set_tolerance(double value)
  {
    if (value >= 0 && value <= 1.0)
      {
        if (value == 0 && method != Truncation::DO_NOT_TRUNCATE)
          {
            method = Truncation::ABSOLUTE_SINGULAR_VALUE;
            truncation_function = select_truncation_function();
          }
        tolerance = value;
        return *this;
      }
    throw std::invalid_argument("Invalid Strategy tolerance");
  }

  Strategy&
  set_simplification_tolerance(double value)
  {
    if (value >= 0 && value <= 1.0)
      {
        simplification_tolerance = value;
        return *this;
      }
    throw std::invalid_argument("Invalid Strategy simplification tolerance");
  }

  Strategy&
  set_max_bond_dimension(size_t value)
  {
    max_bond_dimension = value;
    return *this;
  }

  Strategy&
  set_max_sweeps(int value)
  {
    if (value > 0)
      {
        max_sweeps = value;
        return *this;
      }
    throw std::invalid_argument("Invalid Strategy maximum number of sweeps");
  }

  py::str str() const;

private:
  py::str truncation_name() const;
  py::str simplification_name() const;
};

double destructively_truncate_vector(const py::object a, const Strategy& s);

} // namespace seemps
