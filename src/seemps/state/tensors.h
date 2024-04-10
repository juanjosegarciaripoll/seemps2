#pragma once
#include <limits>
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

extern void (*dgemm_ptr)(char *, char *, int *, int *, int *, double *,
                         double *, int *, double *, int *, double *, double *,
                         int *);
extern void (*zgemm_ptr)(char *, char *, int *, int *, int *,
                         std::complex<double> *, std::complex<double> *, int *,
                         std::complex<double> *, int *, std::complex<double> *,
                         std::complex<double> *, int *);

enum Gemm { GEMM_NORMAL = 0, GEMM_TRANSPOSE = 1, GEMM_ADJOINT = 2 };

py::object gemm(py::object A, Gemm AT, py::object B, Gemm BT);

void load_scipy_wrappers();

/*
 * Numpy structures
 */

inline PyArrayObject *to_array(const py::object &a) {
  return reinterpret_cast<PyArrayObject *>(a.ptr());
}

inline auto is_array(const py::object &a) {
  return static_cast<bool>(PyArray_Check(a.ptr()) != 0);
}

inline auto array_size(const py::object &a) {
  return static_cast<size_t>(PyArray_SIZE(to_array(a)));
}

inline auto array_type(const py::object &a) {
  return PyArray_TYPE(to_array(a));
}

inline auto array_ndim(const py::object &a) {
  return PyArray_NDIM(to_array(a));
}

inline auto array_dim(const py::object &a, int n) {
  return PyArray_DIM(to_array(a), n);
}

inline int array_int_dim(const py::object &a, int n) {
  auto d = array_dim(a, n);
  if (d > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Too large matrix for BLAS/LAPACK library");
  }
  return static_cast<int>(d);
}

inline auto array_dims(const py::object &a) {
  return PyArray_DIMS(to_array(a));
}

inline py::object array_cast(const py::object &a, int type) {
  return py::reinterpret_steal<py::object>(
      reinterpret_cast<PyObject *>(PyArray_Cast(to_array(a), type)));
}

inline py::object array_getcontiguous(const py::object &a) {
  return py::reinterpret_steal<py::object>(
      reinterpret_cast<PyObject *>(PyArray_GETCONTIGUOUS(to_array(a))));
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

inline py::object empty_matrix(npy_intp rows, npy_intp cols, int type) {
  const npy_intp dims[2] = {rows, cols};
  return py::reinterpret_steal<py::object>(PyArray_SimpleNew(2, dims, type));
}

inline py::object zero_matrix(npy_intp rows, npy_intp cols,
                              int type = NPY_DOUBLE) {
  const npy_intp dims[2] = {rows, cols};
  return py::reinterpret_steal<py::object>(PyArray_ZEROS(2, dims, type, 0));
}

py::object eye(npy_intp rows, npy_intp cols);
inline py::object eye(npy_intp rows) { return eye(rows, rows); }

/*
 * Advanced contractions
 */

py::object _matmul(py::object &A, py::object &B);

py::object contract_last_and_first(py::object A, py::object B);
py::object contract_nrjl_ijk_klm(py::object U, py::object A, py::object B);

} // namespace seemps
