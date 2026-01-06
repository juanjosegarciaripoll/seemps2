#include <memory>
#include "tensors.h"

namespace seemps {

py::object schmidt_weights(py::object A) {
  if (!is_array(A) || array_ndim(A) != 3) {
    throw std::invalid_argument(
        "schmidt_weights() only operates on 3-D tensors");
  }
  auto d1 = array_dim(A, 0);
  auto d2 = array_dim(A, 1);
  auto d3 = array_dim(A, 2);
  auto [U, s, V] =
      destructive_svd(array_reshape(array_copy(A), array_dims_t{d1, d2 * d3}));
  _normalize(array_data<double>(s), array_size(s));
  return s * s;
}

static bool _use_gesdd = false;

void _select_svd_driver(std::string which) {
  if (which == "gesvd") {
    _use_gesdd = false;
  } else if (which == "gesdd") {
    _use_gesdd = true;
  } else {
    std::string message = "Unknown svd driver " + which;
    throw std::exception(message.c_str());
  }
}

// TODO: Add set_svd_driver()

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
  auto work = std::make_unique<double[]>(lwork);

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

static int _dgesdd(double *A, double *U, double *s, double *VT, int m, int n,
                   int r) {
  int lwork, info;
  char jobz = (A == U || A == VT) ? 'O' : 'S';
  double work_temp;
  auto iwork = std::make_unique<int>(8 * r);

  // Ask for an estimate of temporary storage needed
  lwork = -1;
  dgesdd_ptr(&jobz, &m, &n, A, &m, s, U, &m, VT, &r, &work_temp, &lwork,
             iwork.get(), &info);
  if (info != 0) {
    return info;
  }

  // Create space in memory for temporary array
  lwork = int(work_temp);
  auto work = std::make_unique<double[]>(lwork);

  // Perform computation using LAPACK
  dgesdd_ptr(&jobz, &m, &n, A, &m, s, U, &m, VT, &r, work.get(), &lwork,
             iwork.get(), &info);
  return info;
}

static int _zgesdd(std::complex<double> *A, std::complex<double> *U, double *s,
                   std::complex<double> *VT, int m, int n, int r) {
  int lwork, info;
  char jobz = (A == U || A == VT) ? 'O' : 'S';
  int lrwork = r * std::max(5 * r + 7, 2 * std::max(m, n) + 2 * r + 1);
  std::complex<double> work_temp;
  auto iwork = std::make_unique<int>(8 * r);
  auto rwork = std::make_unique<double[]>(lrwork);

  // Ask for an estimate of temporary storage needed
  lwork = -1;
  zgesdd_ptr(&jobz, &m, &n, A, &m, s, U, &m, VT, &r, &work_temp, &lwork,
             rwork.get(), iwork.get(), &info);
  if (info != 0) {
    return info;
  }

  lwork = static_cast<int>(work_temp.real());
  auto work = std::make_unique<std::complex<double>[]>(lwork);

  zgesdd_ptr(&jobz, &m, &n, A, &m, s, U, &m, VT, &r, work.get(), &lwork,
             rwork.get(), iwork.get(), &info);
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
    VT = empty_matrix(n, r, type);
    U = A;
  } else {
    U = empty_matrix(r, m, type);
    VT = A;
  }
  s = empty_vector(r, NPY_DOUBLE);
  switch (type) {
  case NPY_DOUBLE:
    err = (_use_gesdd ? _dgesdd : _dgesvd)(
        array_data<double>(A), array_data<double>(U), array_data<double>(s),
        array_data<double>(VT), m, n, r);
    break;
  case NPY_COMPLEX128:
    err = (_use_gesdd ? _zgesdd : _zgesvd)(
        array_data<std::complex<double>>(A),
        array_data<std::complex<double>>(U), array_data<double>(s),
        array_data<std::complex<double>>(VT), m, n, r);
    break;
  case NPY_COMPLEX64:
    return destructive_svd(array_cast(A, NPY_COMPLEX128));
  default:
    return destructive_svd(array_cast(A, NPY_DOUBLE));
  }
  if (err == 0) {
    return {std::move(VT), std::move(s), std::move(U)};
  } else if (err < 0) {
    throw std::runtime_error("Wrong argument passed to LAPACK SVD.");
  } else {
    throw std::runtime_error("SVD did not converge");
  }
}

} // namespace seemps
