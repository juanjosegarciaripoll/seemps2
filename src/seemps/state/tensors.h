#pragma once
#include "core.h"
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>

namespace seemps {

/*
 * BLAS-like operations
 */

double _norm(const double *data, size_t size);

void _normalize(double *data, size_t size, double norm);

void _normalize(double *data, size_t size);

/*
 * Numpy structures
 */

inline PyArrayObject *to_array(const py::object &a) {
  return reinterpret_cast<PyArrayObject *>(a.ptr());
}

inline auto array_size(const py::object &a) {
  return static_cast<size_t>(PyArray_SIZE(to_array(a)));
}

inline auto array_ndim(const py::object &a) {
  return PyArray_NDIM(to_array(a));
}

inline auto array_dim(const py::object &a, int n) {
  return PyArray_DIM(to_array(a), n);
}

inline auto array_dims(const py::object &a) {
  return PyArray_DIMS(to_array(a));
}

inline auto vector_resize_in_place(const py::object &a, size_t new_size) {
  array_dims(a)[0] = new_size;
}

template <typename elt> inline elt *array_data(const py::object &a) {
  return const_cast<elt *>(static_cast<elt *>(PyArray_DATA(to_array(a))));
}

template <class Dimensions>
inline py::object array_reshape(const py::object &a, Dimensions &d) {
  PyArray_Dims dims = {&d[0], static_cast<int>(std::size(d))};
  return py::reinterpret_steal<py::object>(
      PyArray_Newshape(to_array(a), &dims, NPY_CORDER));
}

inline auto as_matrix(const py::object &a, npy_intp rows, npy_intp cols) {
  npy_intp d[2] = {rows, cols};
  return array_reshape(a, d);
}

inline auto matrix_product(const py::object &a, const py::object &b) {
  return py::reinterpret_steal<py::object>(
      PyArray_MatrixProduct(a.ptr(), b.ptr()));
}

/*
 * Advanced contractions
 */

extern py::object numpy, _matmul;

py::object contract_last_and_first(py::object A, py::object B);
py::object contract_nrjl_ijk_klm(py::object U, py::object A, py::object B);

} // namespace seemps
