#include <memory>
#include "tensors.h"

namespace seemps {

static int _dgesvd(double *A, double *U, double *s, double *VT, int m, int n,
                   int r) {
  int lwork, info;
  char jobu = (A == U) ? 'O' : 'S';
  char jobvt = (A == VT) ? 'O' : 'S';
  double work_temp;

  // Ask for an estimate of temporary storage needed
  lwork = -1;
  dgesvd_ptr(&jobu, &jobvt, &m, &n, A, &m, s, U, &m, VT, &r, &work_temp, &lwork,
             &info);
  if (info != 0) {
    return info;
  }

  // Create space in memory for temporary array
  lwork = int(work_temp);
  auto work = std::make_unique<double[]>(static_cast<int>(work_temp));

  // Perform computation using LAPACK
  dgesvd_ptr(&jobu, &jobvt, &m, &n, A, &m, s, U, &m, VT, &r, work.get(), &lwork,
             &info);
  return info;
}

static int _zgesvd(std::complex<double> *A, std::complex<double> *U, double *s,
                   std::complex<double> *VT, int m, int n, int r) {
  int lwork, info;
  char jobu = (A == U) ? 'O' : 'S';
  char jobvt = (A == VT) ? 'O' : 'S';
  std::complex<double> work_temp;

  // Ask for an estimate of temporary storage needed
  lwork = -1;
  zgesvd_ptr(&jobu, &jobvt, &m, &n, A, &m, s, U, &m, VT, &r, &work_temp, &lwork,
             NULL, &info);
  if (info != 0) {
    return info;
  }

  lwork = static_cast<int>(work_temp.real());
  auto work = std::make_unique<std::complex<double>[]>(lwork);
  auto rwork = std::make_unique<double[]>(5 * r);

  zgesvd_ptr(&jobu, &jobvt, &m, &n, A, &m, s, U, &m, VT, &r, work.get(), &lwork,
             rwork.get(), &info);
  return info;
}

/*
 * In Fortran, we have the singular value decomposition
 *
 * A'[m,n] = U'[m,r] s[r] VT'[r,n]
 *
 * where r = min(m,n). In C storage, A[n,m] is the corresponding matrix.
 * Hence, we can write
 *
 *  A[n,m] = VT[n,r] s[r] U[r,m]
 *
 * and the VT and U matrices are actually the usual U and VT from Numpy.
 */
std::tuple<py::object, py::object, py::object> destructive_svd(py::object A) {
  if (!is_array(A) || array_ndim(A) != 2) {
    throw std::invalid_argument("Invalid argument to svd()");
  }
  int m = array_int_dim(A, 1);
  int n = array_int_dim(A, 0);
  auto r = std::min(m, n);
  auto type = array_type(A);
  int err;
  py::object U, s, VT;
  A = array_getcontiguous(A);
  if (r == n) {
    // U matrix is destructively overwritten into A
    VT = zero_matrix(n, r, type);
    U = A;
  } else {
    U = zero_matrix(r, m, type);
    VT = A;
  }
  s = zero_vector(r, NPY_DOUBLE);
  switch (type) {
  case NPY_DOUBLE:
    err = _dgesvd(array_data<double>(A), array_data<double>(U),
                  array_data<double>(s), array_data<double>(VT), m, n, r);
    break;
  case NPY_COMPLEX128:
    err = _zgesvd(array_data<std::complex<double>>(A),
                  array_data<std::complex<double>>(U), array_data<double>(s),
                  array_data<std::complex<double>>(VT), m, n, r);
    break;
  case NPY_COMPLEX64:
    return destructive_svd(array_cast(A, NPY_COMPLEX128));
  default:
    return destructive_svd(array_cast(A, NPY_DOUBLE));
  }
  if (err == 0) {
    return {VT, s, U};
  } else if (err < 0) {
    throw std::runtime_error("Wrong argument passed to LAPACK SVD.");
  } else {
    throw std::runtime_error("SVD did not converge");
  }
}

} // namespace seemps
