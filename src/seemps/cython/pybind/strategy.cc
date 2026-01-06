#include <iostream>
#include "core.h"
#include "tensors.h"
#include "strategy.h"

namespace seemps {

static double _truncate_relative_norm_squared(const py::object &a,
                                              const Strategy &s) {
  static std::vector<double> buffer(1024, 0.0);
  size_t N = array_size(a);
  buffer.resize(N + 1);

  double *errors = &buffer[0];
  double total = 0.0;
  double *data_start = array_data<double>(a);
  double *data = data_start + (N - 1);
  size_t i;
  for (i = 0; i < N; i++, data--) {
    errors[i] = total;
    total += data[0] * data[0];
  }
  errors[N] = total;

  double max_error = total * s.get_tolerance();
  for (i = 1; (i < N) && (errors[i] <= max_error); i++) {
  }
  size_t final_size = std::min(N - i + 1, s.get_max_bond_dimension());
  max_error = errors[N - final_size];
  /*
  if (s.get_normalize_flag()) {
    _normalize(data_start, final_size, std::sqrt(total - max_error));
  }
  */
  if (final_size < N) {
    vector_resize_in_place(a, final_size);
  }
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
  /*
  if (s.get_normalize_flag()) {
    _normalize(data, final_size);
  }
  */
  if (final_size < N) {
    vector_resize_in_place(a, final_size);
  }
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

static py::object contract_nrjl_ijk_klm(py::object U, py::object A,
                                        py::object B) {
  if (PyArray_Check(A.ptr()) == 0 || PyArray_Check(B.ptr()) == 0 ||
      PyArray_Check(U.ptr()) == 0 || array_ndim(A) != 3 || array_ndim(B) != 3 ||
      array_ndim(U) != 2) {
    throw std::invalid_argument("Invalid arguments to _contract_nrjl_ijk_klm");
  }
  auto a = array_dim(A, 0);
  auto d = array_dim(A, 1);
  auto b = array_dim(A, 2);
  auto e = array_dim(B, 1);
  auto c = array_dim(B, 2);
  npy_intp final_dims[4] = {a, d, e, c};
  npy_intp intermediate_dims[3] = {a, d * e, c};
  auto AB = matrix_product(as_matrix(A, a * d, b), as_matrix(B, b, e * c));
  return array_reshape(_matmul(U, array_reshape(AB, intermediate_dims)),
                       final_dims);
}

} // namespace seemps
