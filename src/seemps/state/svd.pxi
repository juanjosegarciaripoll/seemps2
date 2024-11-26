from scipy.linalg.cython_lapack cimport dgesvd, zgesvd, dgesdd, zgesdd
from scipy.linalg import LinAlgError

"""
In Fortran, we have the singular value decomposition

   A'[m,n] = U'[m,r] s[r] VT'[r,n]

where r = min(m,n). In C storage, A[n,m] is the corresponding matrix.
Hence, we can write

   A[n,m] = VT[n,r] s[r] U[r,m]

and the VT and U matrices are actually the usual U and VT from Numpy.
"""

def _destructive_svd(cnp.ndarray A) -> tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray]:
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_NDIM(A) != 2):
        raise ValueError("Invalid argument to SVD")
    return __svd(A)

cdef bint __use_gesdd = 1

def _select_svd_driver(name: str):
    global __use_gesdd
    if name == "gesvd":
        __use_gesdd = 0
    elif name == "gesdd":
        __use_gesdd = 1
    else:
        raise Exception(f"Invalid LAPACK SVD driver name: {name}")

# TODO: Allow selecting other drivers
cdef tuple[cnp.ndarray, cnp.ndarray, cnp.ndarray] __svd(cnp.ndarray A):
    global __use_gesdd
    cdef:
        int m = cnp.PyArray_DIM(A, 1)
        int n = cnp.PyArray_DIM(A, 0)
        int r = min(m, n)
        int type = cnp.PyArray_TYPE(A)
        int err
        cnp.ndarray U, VT
        cnp.ndarray s = _empty_vector(r, cnp.NPY_DOUBLE)
    A = <cnp.ndarray>cnp.PyArray_GETCONTIGUOUS(A)
    if r == n:
        VT = _empty_matrix(n, r, type)
        U = A
    else:
        U = _empty_matrix(r, m, type)
        VT = A
    if type == cnp.NPY_DOUBLE:
        if __use_gesdd:
            err = __dgesdd(<double*>cnp.PyArray_DATA(A),
                           <double*>cnp.PyArray_DATA(U),
                           <double*>cnp.PyArray_DATA(s),
                           <double*>cnp.PyArray_DATA(VT), m, n, r)
        else:
            err = __dgesvd(<double*>cnp.PyArray_DATA(A),
                           <double*>cnp.PyArray_DATA(U),
                           <double*>cnp.PyArray_DATA(s),
                           <double*>cnp.PyArray_DATA(VT), m, n, r)
    elif type == cnp.NPY_COMPLEX128:
        if __use_gesdd:
            err = __zgesdd(<double complex*>cnp.PyArray_DATA(A),
                           <double complex*>cnp.PyArray_DATA(U),
                           <double*>cnp.PyArray_DATA(s),
                           <double complex*>cnp.PyArray_DATA(VT), m, n, r)
        else:
            err = __zgesvd(<double complex*>cnp.PyArray_DATA(A),
                           <double complex*>cnp.PyArray_DATA(U),
                           <double*>cnp.PyArray_DATA(s),
                           <double complex*>cnp.PyArray_DATA(VT), m, n, r)
    elif type == cnp.NPY_COMPLEX64:
        return __svd(<cnp.ndarray>cnp.PyArray_Cast(A, cnp.NPY_COMPLEX128))
    else:
        return __svd(<cnp.ndarray>cnp.PyArray_Cast(A, cnp.NPY_DOUBLE))
    if err == 0:
        return VT, s, U
    elif err < 0:
        raise Exception(f"Wrong argument {-err} to LAPACK SVD.")
    else:
        raise LinAlgError("SVD did not converge")


"""
cdef void dgesvd(
	char *jobu, char *jobvt,
	int *m, int *n, d *a, int *lda,
	d *s, d *u, int *ldu, d *vt, int *ldvt,
	d *work, int *lwork, int *info)
"""
cdef int __dgesvd(double *A, double *U, double *s, double *VT,
                  int m, int n, int r) noexcept:
    cdef:
        int lwork, info
        char *jobu = 'O' if A == U else 'S'
        char *jobvt = 'O' if A == VT else 'S'
        double work_temp
    lwork = -1
    dgesvd(jobu, jobvt,
           &m, &n, A, &m, s, U, &m, VT, &r,
           &work_temp, &lwork, &info)
    if info != 0:
        return info
    lwork = int(work_temp)
    cdef:
        cnp.ndarray work = _empty_vector(lwork, cnp.NPY_DOUBLE)
    dgesvd(jobu, jobvt,
           &m, &n, A, &m, s, U, &m, VT, &r,
           <double*>cnp.PyArray_DATA(work), &lwork, &info)
    return info

"""
void BLAS_FUNC(zgesvd)(
     char *jobu, char *jobvt,
     int *m, int *n, npy_complex128 *a, int *lda,
     double *s, npy_complex128 *u, int *ldu, npy_complex128 *vt, int *ldvt,
     npy_complex128 *work, int *lwork, double *rwork, int *info);
"""

cdef int __zgesvd(double complex*A, double complex*U, double *s, double complex*VT,
                  int m, int n, int r) noexcept:
    cdef:
        int lwork, info
        char *jobu = 'O' if A == U else 'S'
        char *jobvt = 'O' if A == VT else 'S'
        double complex work_temp
        cnp.ndarray rwork = _empty_vector(5 * r, cnp.NPY_DOUBLE)
    lwork = -1
    zgesvd(jobu, jobvt,
           &m, &n, A, &m, s, U, &m, VT, &r,
           &work_temp, &lwork, <double*>cnp.PyArray_DATA(rwork),
           &info)
    if info != 0:
        return info
    lwork = int(work_temp.real)
    cdef:
        cnp.ndarray work = _empty_vector(lwork, cnp.NPY_COMPLEX128)
    zgesvd(jobu, jobvt,
           &m, &n, A, &m, s, U, &m, VT, &r,
           <double complex*>cnp.PyArray_DATA(work), &lwork,
           <double*>cnp.PyArray_DATA(rwork),
           &info)
    return info

"""
void BLAS_FUNC(dgesdd)(
    char *jobz, int *m, int *n, double *a, int *lda,
    double *s, double *u, int *ldu, double *vt, int *ldvt,
    double *work, int *lwork, int *iwork, int *info);
"""

cdef int __dgesdd(double *A, double *U, double *s, double *VT,
                  int m, int n, int r) noexcept:
    cdef:
        int lwork, info
        char *jobz = 'O'
        double work_temp
        cnp.ndarray iwork = _empty_vector(8 * r, cnp.NPY_INT)
    lwork = -1
    dgesdd(jobz,
           &m, &n, A, &m, s, U, &m, VT, &r,
           &work_temp, &lwork,
           <int*>cnp.PyArray_DATA(iwork), &info)
    if info != 0:
        return info
    lwork = int(work_temp)
    cdef:
        cnp.ndarray work = _empty_vector(lwork, cnp.NPY_DOUBLE)
    dgesdd(jobz,
           &m, &n, A, &m, s, U, &m, VT, &r,
           <double*>cnp.PyArray_DATA(work), &lwork,
           <int*>cnp.PyArray_DATA(iwork), &info)
    return info

"""
void BLAS_FUNC(zgesdd)(
  char *jobz, int *m, int *n,
  npy_complex128 *a, int *lda,
  double *s, npy_complex128 *u, int *ldu,
  npy_complex128 *vt, int *ldvt,
  npy_complex128 *work, int *lwork, double *rwork, int *iwork, int *info);
"""

cdef int __zgesdd(double complex*A, double complex*U, double *s, double complex*VT,
                  int m, int n, int r) noexcept:
    cdef:
        int lwork, info
        char *jobz
        double complex work_temp
        int lrwork = r * max(5*r+7, 2*max(m,n)+2*r+1)
        cnp.ndarray rwork = _empty_vector(lrwork, cnp.NPY_DOUBLE)
        cnp.ndarray iwork = _empty_vector(8 * r, cnp.NPY_INT)
    lwork = -1
    if A == U or A == VT:
        jobz = 'O'
    else:
        jobz = 'S'
    zgesdd(jobz,
           &m, &n, A, &m, s, U, &m, VT, &r,
           &work_temp, &lwork, <double*>cnp.PyArray_DATA(rwork),
           <int*>cnp.PyArray_DATA(iwork),
           &info)
    if info != 0:
        return info
    lwork = int(work_temp.real)
    cdef:
        cnp.ndarray work = _empty_vector(lwork, cnp.NPY_COMPLEX128)
    zgesdd(jobz,
           &m, &n, A, &m, s, U, &m, VT, &r,
           <double complex*>cnp.PyArray_DATA(work), &lwork,
           <double*>cnp.PyArray_DATA(rwork),
           <int*>cnp.PyArray_DATA(iwork),
           &info)
    return info
