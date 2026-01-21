#include "strategy.h"
#include "core.h"
#include "tensors.h"
#include <iostream>

namespace seemps
{

// Static member function implementations
double
Strategy::truncate_do_not_truncate(const py::object& a, const Strategy& s)
{
  return 0.0;
}

double
Strategy::truncate_relative_norm_squared(const py::object& a, const Strategy& s)
{
  static std::vector<double> buffer(1024, 0.0);
  size_t N = array_size(a);
  buffer.resize(N + 1);

  double* errors = &buffer[0];
  double total = 0.0;
  double* data_start = array_data<double>(a);
  double* data = data_start + (N - 1);
  size_t i;
  for (i = 0; i < N; i++, data--)
    {
      errors[i] = total;
      total += data[0] * data[0];
    }
  errors[N] = total;

  double max_error = total * s.get_tolerance();
  for (i = 1; (i < N) && (errors[i] <= max_error); i++)
    {
    }
  size_t final_size = std::min(N - i + 1, s.get_max_bond_dimension());
  max_error = errors[N - final_size];
  /*
  if (s.get_normalize_flag()) {
    _normalize(data_start, final_size, std::sqrt(total - max_error));
  }
  */
  if (final_size < N)
    {
      vector_resize_in_place(a, final_size);
    }
  return max_error;
}

double
Strategy::truncate_absolute_singular_value(const py::object& a,
                                           const Strategy& s)
{
  double* data = array_data<double>(a);
  size_t N = seemps::array_size(a);
  double max_error = s.get_tolerance();

  size_t final_size = N;
  for (size_t i = 0; i < N; ++i)
    {
      if (data[i] <= max_error)
        {
          final_size = i;
          break;
        }
    }
  if (final_size == 0)
    {
      final_size = 1;
    }
  else
    {
      final_size = std::min(final_size, s.get_max_bond_dimension());
    }
  max_error = 0.0;
  for (size_t i = final_size; i < N; ++i)
    {
      max_error += data[i] * data[i];
    }
  /*
  if (s.get_normalize_flag()) {
    _normalize(data, final_size);
  }
  */
  if (final_size < N)
    {
      vector_resize_in_place(a, final_size);
    }
  return max_error;
}

double
Strategy::truncate_relative_singular_value(const py::object& a,
                                           const Strategy& s)
{
  double* data = array_data<double>(a);
  size_t N = seemps::array_size(a);
  double max_error = s.get_tolerance() * data[0];

  size_t final_size = N;
  for (size_t i = 0; i < N; ++i)
    {
      if (data[i] <= max_error)
        {
          final_size = i;
          break;
        }
    }
  if (final_size == 0)
    {
      final_size = 1;
    }
  else
    {
      final_size = std::min(final_size, s.get_max_bond_dimension());
    }
  max_error = 0.0;
  for (size_t i = final_size; i < N; ++i)
    {
      max_error += data[i] * data[i];
    }
  /*
  if (s.get_normalize_flag()) {
    _normalize(data, final_size);
  }
  */
  if (final_size < N)
    {
      vector_resize_in_place(a, final_size);
    }
  return max_error;
}

// Helper function to select the appropriate truncation function
Strategy::TruncationFunction
Strategy::select_truncation_function() const
{
  switch (method)
    {
    case Truncation::DO_NOT_TRUNCATE:
      return &Strategy::truncate_do_not_truncate;
    case Truncation::RELATIVE_NORM_SQUARED_ERROR:
      return &Strategy::truncate_relative_norm_squared;
    case Truncation::RELATIVE_SINGULAR_VALUE:
      return &Strategy::truncate_relative_singular_value;
    case Truncation::ABSOLUTE_SINGULAR_VALUE:
      return &Strategy::truncate_absolute_singular_value;
    default:
      return &Strategy::truncate_do_not_truncate;
    }
}

Truncation
Strategy::truncation_from_int(int value)
{
  if (value < 0 || value > 3)
    {
      throw std::invalid_argument("Invalid Strategy Truncation");
    }
  return static_cast<Truncation>(value);
}

Simplification
Strategy::simplification_from_int(int value)
{
  if (value < 0 || value > 2)
    {
      throw std::invalid_argument("Invalid Strategy Simplification");
    }
  return static_cast<Simplification>(value);
}

Strategy
Strategy::replace(py::object a_method, py::object a_tolerance,
                  py::object a_simplification_method,
                  py::object a_simplification_tolerance,
                  py::object a_bond_dimension, py::object a_num_sweeps,
                  py::object a_normalize_flag) const
{
  Strategy output = *this;
  if (!a_method.is_none())
    {
      output.method = truncation_from_int(py::int_(a_method));
      output.truncation_function = output.select_truncation_function();
    }
  if (!a_tolerance.is_none())
    {
      output.set_tolerance(py::cast<double>(a_tolerance));
    }
  if (!a_simplification_method.is_none())
    {
      output.set_simplification_method(
          simplification_from_int(py::int_(a_simplification_method)));
    }
  if (!a_simplification_tolerance.is_none())
    {
      output.set_simplification_tolerance(
          py::float_(a_simplification_tolerance));
    }
  if (!a_bond_dimension.is_none())
    {
      output.set_max_bond_dimension(py::int_(a_bond_dimension));
    }
  if (!a_num_sweeps.is_none())
    {
      output.set_max_sweeps(py::int_(a_num_sweeps));
    }
  if (!a_normalize_flag.is_none())
    {
      output.normalize = py::cast<bool>(a_normalize_flag);
    }
  return output;
}

py::str
Strategy::str() const
{
  py::str format = ("Strategy(method={method}, tolerance={tolerance:5g}, "
                    "max_bond_dimension={max_bond_dimension}, "
                    "normalize={normalize}, "
                    "simplify={simplification_method}, "
                    "simplification_tolerance={simplification_tolerance:5g}, "
                    "max_sweeps={max_sweeps})");
  return format.format(
      "method"_a = truncation_name(), "tolerance"_a = tolerance,
      "max_bond_dimension"_a = max_bond_dimension, "normalize"_a = normalize,
      "simplification_method"_a = simplification_name(),
      "simplification_tolerance"_a = simplification_tolerance,
      "max_sweeps"_a = max_sweeps);
}

py::str
Strategy::truncation_name() const
{
  switch (method)
    {
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

py::str
Strategy::simplification_name() const
{
  switch (simplification_method)
    {
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

double
destructively_truncate_vector(const py::object a, const Strategy& s)
{
  if (!PyArray_Check(a.ptr()))
    {
      throw std::invalid_argument(
          "destructively_truncate_vector expected an ndarray");
    }
  return s.truncate(a);
}

} // namespace seemps
