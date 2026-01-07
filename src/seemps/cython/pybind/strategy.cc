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
