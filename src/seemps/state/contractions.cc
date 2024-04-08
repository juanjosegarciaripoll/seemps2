#include <iostream>
#include "core.h"
#include "tensors.h"

namespace seemps {

py::object contract_nrjl_ijk_klm(py::object U, py::object A, py::object B) {
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

py::object contract_last_and_first(py::object A, py::object B) {
  if (!PyArray_Check(A.ptr()) || !PyArray_Check(B.ptr())) {
    throw std::invalid_argument("_contract_last_and_first_expect tensors");
  }

  auto ndimA = array_ndim(A);
  auto Alast = array_dim(A, ndimA - 1);
  auto ndimB = array_ndim(B);
  auto Bfirst = array_dim(B, 0);
  std::vector<npy_intp> new_dims;
  new_dims.reserve(ndimA + ndimB - 2);
  for (int i = 0; i < ndimA - 1; ++i) {
    new_dims.emplace_back(array_dim(A, i));
  }
  for (int i = 1; i < ndimB; ++i) {
    new_dims.emplace_back(array_dim(B, i));
  }
  auto C = matrix_product(as_matrix(A, array_size(A) / Alast, Alast),
                          as_matrix(B, Bfirst, array_size(B) / Bfirst));
  return array_reshape(C, new_dims);
}

} // namespace seemps
