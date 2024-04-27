#include <iostream>
#include "tensors.h"

namespace seemps {

std::complex<double> array_vdot(const py::object &someA,
                                const py::object &someB) {
  py::object A = array_getcontiguous(someA);
  py::object B = array_getcontiguous(someB);
  int one = 1;
  int L = array_size(A);
  if (array_size(B) != L) {
    throw std::invalid_argument("Arrays of different size in vdot");
  }
  switch (array_type(A)) {
  case NPY_DOUBLE:
    switch (array_type(B)) {
    case NPY_DOUBLE:
      return ddot_ptr(&L, array_data<double>(A), &one, array_data<double>(B),
                      &one);
    case NPY_COMPLEX64:
    case NPY_COMPLEX128:
      return array_vdot(array_cast(A, NPY_COMPLEX128), B);
    default:
      return array_vdot(A, array_cast(B, NPY_DOUBLE));
    }
  case NPY_COMPLEX64:
    A = array_cast(A, NPY_COMPLEX128);
  case NPY_COMPLEX128:
    switch (array_type(B)) {
    case NPY_COMPLEX128:
      return zdotc_ptr(&L, array_data<std::complex<double>>(A), &one,
                       array_data<std::complex<double>>(B), &one);
    default:
      return array_vdot(A, array_cast(B, NPY_COMPLEX128));
    }
  default:
    // A is real
    switch (array_type(B)) {
    case NPY_COMPLEX64:
      B = array_cast(B, NPY_COMPLEX128);
    case NPY_COMPLEX128:
      return array_vdot(array_cast(A, NPY_COMPLEX128), B);
    default:
      return array_vdot(array_cast(A, NPY_DOUBLE), array_cast(B, NPY_DOUBLE));
    }
  }
}

/*
 * Equivalent to A[:rows,:cols]
 */
py::object matrix_resize(py::object A, npy_intp rows, npy_intp cols) {
  auto old_rows = array_dim(A, 0);
  auto old_cols = array_dim(A, 1);
  if (rows < 0) {
    rows = old_rows;
  }
  if (cols < 0) {
    cols = old_cols;
  }
  if (rows == old_rows && cols == old_cols) {
    return A;
  }
  auto dims = array_dims(A);
  std::swap(dims[0], rows);
  std::swap(dims[1], cols);
  auto output = array_copy(A);
  std::swap(dims[0], rows);
  std::swap(dims[1], cols);
  return output;
}

/*
 * BLAS Operations
 */

double _norm(const double *data, size_t size) {
  double output = 0.0;
  for (; size; --size, ++data) {
    output += (*data) * (*data);
  }
  return std::sqrt(output);
}

void _normalize(double *data, size_t size, double norm) {
  for (; size; --size, ++data) {
    *data /= norm;
  }
}

void _normalize(double *data, size_t size) {
  return _normalize(data, size, _norm(data, size));
}

/*
 * Matrix multiplication
 *
 * In Python we use C-stype order for arrays. In BLAS, the arrays are
 * in Fortran order. Thus, a contraction in Python
 *     C[n,m] = B[n,k] * A[k,m]
 * is equivalent to *gemm(B', A', C')
 *     C'[m,n] = A'[m,k] * B'[k,n]
 * where A', B' and C' are Fotran ordered arrays on the same memory region.
 */

static py::object _dgemm(const py::object &A, const py::object &B, int m, int n,
                         int k, char Aorder, char Border) {
  auto C = empty_matrix(n, m, NPY_DOUBLE);
  int lda = array_int_dim(A, 1);
  int ldb = array_int_dim(B, 1);
  double alpha = 1.0;
  double beta = 0.0;
  if (Border == 'C') {
    Border = 'T';
  }
  if (Aorder == 'C') {
    Aorder = 'T';
  }
  dgemm_ptr(&Aorder, &Border, &m, &n, &k, &alpha, array_data<double>(A), &lda,
            array_data<double>(B), &ldb, &beta, array_data<double>(C), &m);
  return C;
}

static py::object _zgemm(const py::object &A, const py::object &B, int m, int n,
                         int k, char Aorder, char Border) {
  auto C = empty_matrix(n, m, NPY_COMPLEX128);
  int lda = array_int_dim(A, 1);
  int ldb = array_int_dim(B, 1);
  std::complex<double> alpha = 1.0;
  std::complex<double> beta = 0.0;
  zgemm_ptr(&Aorder, &Border, &m, &n, &k, &alpha,
            array_data<std::complex<double>>(A), &lda,
            array_data<std::complex<double>>(B), &ldb, &beta,
            array_data<std::complex<double>>(C), &m);
  return C;
}

py::object gemm(py::object B, Gemm BT, py::object A, Gemm AT) {
  A = array_getcontiguous(A);
  B = array_getcontiguous(B);
  int m, n, k;
  char Aorder, Border;
  if (AT == Gemm::GEMM_NORMAL) {
    m = array_int_dim(A, 1);
    k = array_int_dim(A, 0);
    Aorder = 'N';
  } else {
    m = array_int_dim(A, 0);
    k = array_int_dim(A, 1);
    Aorder = (AT == Gemm::GEMM_TRANSPOSE) ? 'T' : 'C';
  }
  if (BT == Gemm::GEMM_NORMAL) {
    n = array_int_dim(B, 0);
    if (k != array_dim(B, 1)) {
      throw std::invalid_argument("Non-matching matrices in GEMM");
    }
    Border = 'N';
  } else {
    n = array_int_dim(B, 1);
    if (k != array_dim(B, 0)) {
      throw std::invalid_argument("Non-matching matrices in GEMM");
    }
    Border = (BT == Gemm::GEMM_TRANSPOSE) ? 'T' : 'C';
  }
  switch (array_type(A)) {
  case NPY_DOUBLE:
    switch (array_type(B)) {
    case NPY_DOUBLE:
      return _dgemm(A, B, m, n, k, Aorder, Border);
    case NPY_COMPLEX64:
      B = array_cast(B, NPY_COMPLEX128);
    case NPY_COMPLEX128:
      return _zgemm(array_cast(A, NPY_COMPLEX128), B, m, n, k, Aorder, Border);
    default:
      return _dgemm(A, array_cast(B, NPY_DOUBLE), m, n, k, Aorder, Border);
    }
  case NPY_COMPLEX128:
    return _zgemm(
        A, array_type(B) == NPY_COMPLEX128 ? B : array_cast(B, NPY_COMPLEX128),
        m, n, k, Aorder, Border);
  case NPY_COMPLEX64:
    return _zgemm(array_cast(A, NPY_COMPLEX128), array_cast(B, NPY_COMPLEX128),
                  m, n, k, Aorder, Border);
  default:
    return _dgemm(array_cast(A, NPY_DOUBLE), array_cast(B, NPY_DOUBLE), m, n, k,
                  Aorder, Border);
  }
}
} // namespace seemps
