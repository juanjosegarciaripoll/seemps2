#pragma once
#include "core.h"
#include <limits>
#include <numpy/arrayobject.h>
#include <pybind11/numpy.h>
#include <tuple>

namespace seemps
{
/*
 * BLAS-like operations
 */

double _norm(const double* data, size_t size);

void _normalize(double* data, size_t size, double norm);

void _normalize(double* data, size_t size);

extern void (*dgemm_ptr)(char*, char*, int*, int*, int*, double*, double*, int*,
                         double*, int*, double*, double*, int*);
extern void (*zgemm_ptr)(char*, char*, int*, int*, int*, std::complex<double>*,
                         std::complex<double>*, int*, std::complex<double>*,
                         int*, std::complex<double>*, std::complex<double>*,
                         int*);

extern void (*dgesvd_ptr)(char* jobu, char* jobvt, int* m, int* n, double* a,
                          int* lda, double* s, double* u, int* ldu, double* vt,
                          int* ldvt, double* work, int* lwork, int* info);
extern void (*zgesvd_ptr)(char* jobu, char* jobvt, int* m, int* n,
                          std::complex<double>* a, int* lda, double* s,
                          std::complex<double>* u, int* ldu,
                          std::complex<double>* vt, int* ldvt,
                          std::complex<double>* work, int* lwork, double* rwork,
                          int* info);

extern void (*dgesdd_ptr)(char* jobz, int* m, int* n, double* a, int* lda,
                          double* s, double* u, int* ldu, double* vt, int* ldvt,
                          double* work, int* lwork, int* iwork, int* info);
extern void (*zgesdd_ptr)(char* jobz, int* m, int* n, std::complex<double>* a,
                          int* lda, double* s, std::complex<double>* u,
                          int* ldu, std::complex<double>* vt, int* ldvt,
                          std::complex<double>* work, int* lwork, double* rwork,
                          int* iwork, int* info);

enum Gemm
{
  GEMM_NORMAL = 0,
  GEMM_TRANSPOSE = 1,
  GEMM_ADJOINT = 2
};

py::object gemm(py::object A, Gemm AT, py::object B, Gemm BT);

std::tuple<py::object, py::object, py::object> destructive_svd(py::object A);

void _select_svd_driver(std::string which);

void load_scipy_wrappers();

/*
 * Numpy structures
 */

inline PyArrayObject*
to_array(const py::object& a)
{
  return reinterpret_cast<PyArrayObject*>(a.ptr());
}

inline auto
is_array(const py::object& a)
{
  return static_cast<bool>(PyArray_Check(a.ptr()) != 0);
}

inline auto
array_size(const py::object& a)
{
  return PyArray_SIZE(to_array(a));
}

inline auto
array_type(const py::object& a)
{
  return PyArray_TYPE(to_array(a));
}

inline auto
array_ndim(const py::object& a)
{
  return PyArray_NDIM(to_array(a));
}

inline auto
array_dim(const py::object& a, int n)
{
  return PyArray_DIM(to_array(a), n);
}

inline int
array_int_dim(const py::object& a, int n)
{
  auto d = array_dim(a, n);
  if (d > std::numeric_limits<int>::max())
    {
      throw std::invalid_argument("Too large matrix for BLAS/LAPACK library");
    }
  return static_cast<int>(d);
}

inline auto
array_dims(const py::object& a)
{
  return PyArray_DIMS(to_array(a));
}

inline py::object
array_cast(const py::object& a, int type)
{
  return py::reinterpret_steal<py::object>(
      reinterpret_cast<PyObject*>(PyArray_Cast(to_array(a), type)));
}

inline py::object
array_getcontiguous(const py::object& a)
{
  return py::reinterpret_steal<py::object>(
      reinterpret_cast<PyObject*>(PyArray_GETCONTIGUOUS(to_array(a))));
}

inline auto
vector_resize_in_place(const py::object& a, size_t new_size)
{
  array_dims(a)[0] = new_size;
}

template <typename elt>
inline elt*
array_data(const py::object& a)
{
  return const_cast<elt*>(static_cast<elt*>(PyArray_DATA(to_array(a))));
}

using array_dims_t = std::initializer_list<npy_intp>;

template <class Dimensions>
inline py::object
array_reshape(const py::object& a, const Dimensions& d)
{
  PyArray_Dims dims = { const_cast<npy_intp*>(&(*std::begin(d))),
                        static_cast<int>(std::size(d)) };
  return py::reinterpret_steal<py::object>(
      PyArray_Newshape(to_array(a), &dims, NPY_CORDER));
}

// TODO: Make this more general
py::object matrix_resize(py::object A, npy_intp rows, npy_intp cols);

inline auto
as_matrix(const py::object& a, npy_intp rows, npy_intp cols)
{
  npy_intp d[2] = { rows, cols };
  return array_reshape(a, d);
}

inline auto
as_3tensor(const py::object& A, npy_intp a, npy_intp b, npy_intp c)
{
  npy_intp d[3] = { a, b, c };
  return array_reshape(A, d);
}

inline auto
array_copy(const py::object& A)
{
  return py::reinterpret_steal<py::object>(
      PyArray_FROM_OF(A.ptr(), NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
                                   | NPY_ARRAY_ENSURECOPY));
}

inline auto
matrix_product(const py::object& a, const py::object& b)
{
  return py::reinterpret_steal<py::object>(
      PyArray_MatrixProduct(a.ptr(), b.ptr()));
}

inline py::object
empty_vector(npy_intp size, int type)
{
  const npy_intp dims[1] = { size };
  return py::reinterpret_steal<py::object>(PyArray_SimpleNew(1, dims, type));
}

inline py::object
empty_matrix(npy_intp rows, npy_intp cols, int type)
{
  const npy_intp dims[2] = { rows, cols };
  return py::reinterpret_steal<py::object>(PyArray_SimpleNew(2, dims, type));
}

inline py::object
zero_matrix(npy_intp rows, npy_intp cols, int type = NPY_DOUBLE)
{
  const npy_intp dims[2] = { rows, cols };
  return py::reinterpret_steal<py::object>(PyArray_ZEROS(2, dims, type, 0));
}

inline py::object
zero_vector(npy_intp size, int type = NPY_DOUBLE)
{
  const npy_intp dims[1] = { size };
  return py::reinterpret_steal<py::object>(PyArray_ZEROS(1, dims, type, 0));
}

py::object eye(npy_intp rows, npy_intp cols);
inline py::object
eye(npy_intp rows)
{
  return eye(rows, rows);
}

/*
 * Advanced contractions
 */

py::object _matmul(const py::object& A, const py::object& B);

py::object contract_last_and_first(py::object A, py::object B);
py::object contract_nrjl_ijk_klm(py::object U, py::object A, py::object B);

} // namespace seemps
