#include <iostream>
#include "core.h"
#include "tensors.h"

namespace seemps {

py::object _matmul(const py::object &A, const py::object &B) {
  auto numpy = py::module_::import("numpy");
  auto matmul = numpy.attr("matmul");
  return matmul(A, B);
}

py::object contract_nrjl_ijk_klm(const py::object &U, const py::object &A,
                                 const py::object &B) {
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

py::object contract_last_and_first(const py::object &A, const py::object &B) {
  if (!PyArray_Check(A.ptr()) || !PyArray_Check(B.ptr())) {
    throw std::invalid_argument("_contract_last_and_first_expect tensors");
  }

  auto ndimA = array_ndim(A);
  auto Alast = array_dim(A, ndimA - 1);
  auto ndimB = array_ndim(B);
  auto Bfirst = array_dim(B, 0);
  auto ndimC = ndimA + ndimB - 2;
  std::vector<npy_intp> new_dims(ndimC);
  auto Adims = array_dims(A);
  std::copy(Adims, Adims + ndimA, new_dims.begin());
  auto Bdims = array_dims(B);
  std::copy(Bdims + 1, Bdims + ndimB, new_dims.begin() + ndimA - 1);
  return array_reshape(
      matrix_product(as_matrix(A, array_size(A) / Alast, Alast),
                     as_matrix(B, Bfirst, array_size(B) / Bfirst)),
      new_dims);
}

} // namespace seemps
