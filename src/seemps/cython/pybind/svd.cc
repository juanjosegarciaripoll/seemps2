#include <memory>
#include <iostream>
#include "tensors.h"

namespace seemps
{

py::object
schmidt_weights(py::object A)
{
  if (!is_array(A) || array_ndim(A) != 3)
    {
      throw std::invalid_argument(
          "schmidt_weights() only operates on 3-D tensors");
    }
  auto d1 = array_dim(A, 0);
  auto d2 = array_dim(A, 1);
  auto d3 = array_dim(A, 2);
  auto [U, s, V] = svd(as_matrix(A, d1, d2 * d3));
  _normalize(array_data<double>(s), array_size(s));
  return s * s;
}

static bool _use_gesdd = true;

void
_select_svd_driver(std::string which)
{
  if (which == "gesvd")
    {
      _use_gesdd = false;
    }
  else if (which == "gesdd")
    {
      _use_gesdd = true;
    }
  else
    {
      std::string message = "Unknown svd driver " + which;
      throw std::runtime_error(message.c_str());
    }
}

static inline py::object
_coerce_to_supported_svd_type(py::object A)
{
  if (!is_array(A))
    {
      throw std::invalid_argument("Non-array type passed to SVD");
    }
  switch (array_type(A))
    {
    case NPY_DOUBLE:
    case NPY_COMPLEX128:
      return A;
    case NPY_COMPLEX64:
      // std::cerr << "Conversion to COMPLEX128\n";
      return array_cast(A, NPY_COMPLEX128);
    default:
      // std::cerr << "Conversion to DOUBLE\n";
      return array_cast(A, NPY_DOUBLE);
    }
}

SVDData::SVDData(const py::object& orig_A, bool destructive)
    // TODO: allow conly continguous_column_blas_matrix
    : A{ ensure_contiguous_blas_matrix(_coerce_to_supported_svd_type(orig_A)) },
      type{ array_type(A) }, m{ static_cast<int>(array_dim(A, 1)) },
      n{ static_cast<int>(array_dim(A, 0)) }, jobU{ 'S' }, jobVT{ 'S' },
      jobz{ 'S' }, overwrite{ destructive || (to_array(A) != to_array(orig_A)) }
{
  if (overwrite)
    {
      ldA = blas_matrix_leading_dimension(A);
    }
  else
    {
      // A will be destroyed but we don't want it
      A = array_copy(A);
      ldA = m;
    }
  if (m >= n)
    {
      VT = empty_matrix(n, ldVT = r = n, type);
      // U matrix is destructively overwritten into A
      U = A;
      ldU = ldA;
      jobU = jobz = 'O';
    }
  else
    {
      U = empty_matrix(m, ldU = r = m, type);
      // VT matrix is destructively overwritten into A
      VT = A;
      ldVT = ldA;
      jobVT = jobz = 'O';
    }
  s = empty_vector(r, NPY_DOUBLE);
}

std::tuple<py::object, py::object, py::object>
SVDData::run()
{
  if (r == 0)
    {
      return { std::move(VT), std::move(s), std::move(U) };
    }
  int err;
  if (type == NPY_DOUBLE)
    {
      err = _use_gesdd ? _dgesdd() : _dgesvd();
    }
  else
    {
      err = _use_gesdd ? _zgesdd() : _zgesvd();
    }
  if (err == 0)
    {
      return { std::move(VT), std::move(s), std::move(U) };
    }
  if (err < 0)
    {
      throw std::runtime_error("Wrong argument passed to LAPACK SVD.");
    }
  // TODO: Re-run SVD with the other method
  throw std::runtime_error("SVD did not converge");
}

int
SVDData::_dgesvd()
{
  // Ask for an estimate of temporary storage needed
  int lwork = -1;
  double work_temp;
  dgesvd_ptr(&jobU, &jobVT, &m, &n, array_data<double>(A), &ldA,
             array_data<double>(s), array_data<double>(U), &ldU,
             array_data<double>(VT), &ldVT, &work_temp, &lwork, &info);
  if (info != 0)
    {
      return info;
    }

  // Create space in memory for temporary array
  lwork = int(work_temp);
  auto work = std::make_unique<double[]>(lwork);

  // Perform computation using LAPACK
  dgesvd_ptr(&jobU, &jobVT, &m, &n, array_data<double>(A), &ldA,
             array_data<double>(s), array_data<double>(U), &ldU,
             array_data<double>(VT), &ldVT, work.get(), &lwork, &info);
  return info;
}

int
SVDData::_zgesvd()
{
  typedef std::complex<double> complex;
  // Ask for an estimate of temporary storage needed
  int lwork = -1;
  complex work_temp;
  zgesvd_ptr(&jobU, &jobVT, &m, &n, array_data<complex>(A), &ldA,
             array_data<double>(s), array_data<complex>(U), &ldU,
             array_data<complex>(VT), &ldVT, &work_temp, &lwork, NULL, &info);
  if (info != 0)
    {
      return info;
    }

  lwork = static_cast<int>(work_temp.real());
  auto work = std::make_unique<complex[]>(lwork);
  auto rwork = std::make_unique<double[]>(5 * r);

  zgesvd_ptr(&jobU, &jobVT, &m, &n, array_data<complex>(A), &ldA,
             array_data<double>(s), array_data<complex>(U), &ldU,
             array_data<complex>(VT), &ldVT, work.get(), &lwork, rwork.get(),
             &info);
  return info;
}

int
SVDData::_dgesdd()
{
  // Ask for an estimate of temporary storage needed
  int lwork = -1;
  double work_temp;
  auto iwork = std::make_unique<int[]>(8 * r);
  dgesdd_ptr(&jobz, &m, &n, array_data<double>(A), &ldA, array_data<double>(s),
             array_data<double>(U), &ldU, array_data<double>(VT), &ldVT,
             &work_temp, &lwork, iwork.get(), &info);
  if (info != 0)
    {
      return info;
    }

  // Create space in memory for temporary array
  lwork = int(work_temp);
  auto work = std::make_unique<double[]>(lwork);
  // Perform computation using LAPACK
  dgesdd_ptr(&jobz, &m, &n, array_data<double>(A), &ldA, array_data<double>(s),
             array_data<double>(U), &ldU, array_data<double>(VT), &ldVT,
             work.get(), &lwork, iwork.get(), &info);
  return info;
}

int
SVDData::_zgesdd()
{
  typedef std::complex<double> complex;
  int lrwork = r * std::max(5 * r + 7, 2 * std::max(m, n) + 2 * r + 1);
  complex work_temp;
  auto iwork = std::make_unique<int[]>(8 * r);
  auto rwork = std::make_unique<double[]>(lrwork);

  // Ask for an estimate of temporary storage needed
  int lwork = -1;
  zgesdd_ptr(&jobz, &m, &n, array_data<complex>(A), &ldA, array_data<double>(s),
             array_data<complex>(U), &ldU, array_data<complex>(VT), &ldVT,
             &work_temp, &lwork, rwork.get(), iwork.get(), &info);
  if (info != 0)
    {
      return info;
    }

  lwork = static_cast<int>(work_temp.real());
  auto work = std::make_unique<complex[]>(lwork);

  zgesdd_ptr(&jobz, &m, &n, array_data<complex>(A), &ldA, array_data<double>(s),
             array_data<complex>(U), &ldU, array_data<complex>(VT), &ldVT,
             work.get(), &lwork, rwork.get(), iwork.get(), &info);
  return info;
}

} // namespace seemps
