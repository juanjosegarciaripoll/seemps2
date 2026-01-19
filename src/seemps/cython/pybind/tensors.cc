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

template <typename number> class GemmData
{
  typedef number number;

  static py::object
  ensure_contiguous_column(py::object& A)
  {
    check_array_is_blas_compatible(A);
    if (array_stride(A, 1) != sizeof(number))
      {
        // throw std::exception("Non-contiguous array in GEMM");
        // std::cerr << "Non-contiguous array in GEMM\n";
        return array_getcontiguous(A);
      }
    return A;
  }

  GemmData();
  GemmData(const GemmData&) = delete;
  GemmData(const GemmData&&) = delete;

public:
  number alpha = 1.0;
  number beta = 0.0;
  char* Aorder = "N";
  char* Border = "N";
  py::object A;
  int m, k, lda;
  py::object B;
  int kb, n, ldb;

  GemmData(py::object& oA, Gemm AT, py::object& oB, Gemm BT)
      : A{ ensure_contiguous_column(oA) },
        m{ static_cast<int>(array_dim(A, 1)) },
        k{ static_cast<int>(array_dim(A, 0)) },
        lda{ static_cast<int>(array_stride(A, 0) / sizeof(number)) },
        B{ ensure_contiguous_column(oB) },
        kb{ static_cast<int>(array_dim(B, 1)) },
        n{ static_cast<int>(array_dim(B, 0)) },
        ldb{ static_cast<int>(array_stride(B, 0) / sizeof(number)) }
  {
    if (AT != Gemm::GEMM_NORMAL)
      {
        std::swap(m, k);
        Aorder = (AT == Gemm::GEMM_TRANSPOSE) ? "T" : "C";
      }
    if (BT != Gemm::GEMM_NORMAL)
      {
        std::swap(kb, n);
        Border = (BT == Gemm::GEMM_TRANSPOSE) ? "T" : "C";
      }
    if (kb != k)
      {
        throw std::exception("A and B matrices have wrong dimensions in GEMM");
      }
  }

  py::object gemm();
};

template <>
py::object
GemmData<double>::gemm()
{
  auto C = empty_matrix(n, m, NPY_DOUBLE);
  dgemm_ptr(Aorder, Border, &m, &n, &k, &alpha, array_data<number>(A), &lda,
            array_data<number>(B), &ldb, &beta, array_data<number>(C), &m);
  return C;
}

template <>
py::object
GemmData<std::complex<double>>::gemm()
{
  auto C = empty_matrix(n, m, NPY_COMPLEX128);
  zgemm_ptr(Aorder, Border, &m, &n, &k, &alpha, array_data<number>(A), &lda,
            array_data<number>(B), &ldb, &beta, array_data<number>(C), &m);
  return C;
}

py::object
gemm(py::object& B, Gemm BT, py::object& A, Gemm AT)
{
  switch (array_type(A))
    {
    case NPY_DOUBLE:
      switch (array_type(B))
        {
        case NPY_DOUBLE:
          return GemmData<double>(A, AT, B, BT).gemm();
        case NPY_COMPLEX64:
          B = array_cast(B, NPY_COMPLEX128);
        case NPY_COMPLEX128:
          return GemmData<std::complex<double>>(array_cast(A, NPY_COMPLEX128),
                                                AT, B, BT)
              .gemm();
        default:
          return GemmData<double>(A, AT, array_cast(B, NPY_DOUBLE), BT).gemm();
        }
    case NPY_COMPLEX128:
      return GemmData<std::complex<double>>(A, AT,
                                            array_type(B) == NPY_COMPLEX128
                                                ? B
                                                : array_cast(B, NPY_COMPLEX128),
                                            BT)
          .gemm();
    case NPY_COMPLEX64:
      return GemmData<std::complex<double>>(array_cast(A, NPY_COMPLEX128), AT,
                                            array_cast(B, NPY_COMPLEX128), BT)
          .gemm();
    default:
      return GemmData<double>(array_cast(A, NPY_DOUBLE), AT,
                              array_cast(B, NPY_DOUBLE), BT)
          .gemm();
    }
}
} // namespace seemps
