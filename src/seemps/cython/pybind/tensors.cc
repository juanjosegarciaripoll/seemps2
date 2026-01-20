#include <iostream>
#include "tensors.h"

namespace seemps
{

/*
 * Equivalent to A[:rows,:cols]
 */
py::object
matrix_resize(py::object A, npy_intp rows, npy_intp cols)
{
  auto old_rows = array_dim(A, 0);
  auto old_cols = array_dim(A, 1);
  if (rows < 0)
    {
      rows = old_rows;
    }
  if (cols < 0)
    {
      cols = old_cols;
    }
  if (rows == old_rows && cols == old_cols)
    {
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

double
_norm(const double* data, size_t size)
{
  double output = 0.0;
  for (; size; --size, ++data)
    {
      output += (*data) * (*data);
    }
  return std::sqrt(output);
}

void
_normalize(double* data, size_t size, double norm)
{
  for (; size; --size, ++data)
    {
      *data /= norm;
    }
}

void
_normalize(double* data, size_t size)
{
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

class GemmData
{
  static inline char orders[] = { 'N', 'T', 'C' };

  GemmData();
  GemmData(const GemmData&) = delete;
  GemmData(const GemmData&&) = delete;

public:
  const py::object A, B;
  int m, k, lda, kb, n, ldb;
  char Aorder, Border;

  GemmData(const py::object& oA, Gemm AT, const py::object& oB, Gemm BT)
      : A{ ensure_contiguous_column_blas_matrix(oA) },
        B{ ensure_contiguous_column_blas_matrix(oB) },
        m{ static_cast<int>(array_dim(A, 1)) },
        k{ static_cast<int>(array_dim(A, 0)) },
        kb{ static_cast<int>(array_dim(B, 1)) },
        n{ static_cast<int>(array_dim(B, 0)) },
        ldb{ blas_matrix_leading_dimension(B) }, Aorder{ orders[AT] },
        Border{ orders[BT] }
  {
    if (AT != Gemm::GEMM_NORMAL)
      {
        std::swap(m, k);
      }
    if (BT != Gemm::GEMM_NORMAL)
      {
        std::swap(kb, n);
      }
    if (kb != k)
      {
        throw std::logic_error(
            "A and B matrices have wrong dimensions in GEMM");
      }
  }

  py::object
  dgemm()
  {
    auto C = empty_matrix(n, m, NPY_DOUBLE);
    double alpha = 1.0;
    double beta = 0.0;
    int lda = blas_matrix_leading_dimension_from_type<double>(A);
    dgemm_ptr(&Aorder, &Border, &m, &n, &k, &alpha, array_data<double>(A), &lda,
              array_data<double>(B), &ldb, &beta, array_data<double>(C), &m);
    return C;
  }

  py::object
  zgemm()
  {
    auto C = empty_matrix(n, m, NPY_COMPLEX128);
    std::complex<double> alpha = 1.0;
    std::complex<double> beta = 0.0;
    int lda = blas_matrix_leading_dimension_from_type<std::complex<double>>(A);
    zgemm_ptr(&Aorder, &Border, &m, &n, &k, &alpha,
              array_data<std::complex<double>>(A), &lda,
              array_data<std::complex<double>>(B), &ldb, &beta,
              array_data<std::complex<double>>(C), &m);
    return C;
  }
};

py::object
gemm(py::object B, Gemm BT, py::object A, Gemm AT)
{
  switch (array_type(A))
    {
    case NPY_DOUBLE:
      switch (array_type(B))
        {
        case NPY_DOUBLE:
          return GemmData(A, AT, B, BT).dgemm();
        case NPY_COMPLEX64:
          B = array_cast(B, NPY_COMPLEX128);
        case NPY_COMPLEX128:
          return GemmData(array_cast(A, NPY_COMPLEX128), AT, B, BT).zgemm();
        default:
          return GemmData(A, AT, array_cast(B, NPY_DOUBLE), BT).dgemm();
        }
    case NPY_COMPLEX128:
      return GemmData(A, AT,
                      array_type(B) == NPY_COMPLEX128
                          ? B
                          : array_cast(B, NPY_COMPLEX128),
                      BT)
          .zgemm();
    case NPY_COMPLEX64:
      return GemmData(array_cast(A, NPY_COMPLEX128), AT,
                      array_cast(B, NPY_COMPLEX128), BT)
          .zgemm();
    default:
      return GemmData(array_cast(A, NPY_DOUBLE), AT, array_cast(B, NPY_DOUBLE),
                      BT)
          .dgemm();
    }
}
} // namespace seemps
