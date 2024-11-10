#include "core.h"
#include "tensors.h"

namespace seemps {

std::complex<double> (*zdotc_ptr)(int *n, std::complex<double> *zx, int *incx,
                                  std::complex<double> *zy, int *incy);
double (*ddot_ptr)(int *n, double *zx, int *incx, double *zy, int *incy);
void (*dgemm_ptr)(char *, char *, int *, int *, int *, double *, double *,
                  int *, double *, int *, double *, double *, int *);
void (*zgemm_ptr)(char *, char *, int *, int *, int *, std::complex<double> *,
                  std::complex<double> *, int *, std::complex<double> *, int *,
                  std::complex<double> *, std::complex<double> *, int *);
void (*dgesvd_ptr)(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
                   double *s, double *u, int *ldu, double *vt, int *ldvt,
                   double *work, int *lwork, int *info);
void (*zgesvd_ptr)(char *jobu, char *jobvt, int *m, int *n,
                   std::complex<double> *a, int *lda, double *s,
                   std::complex<double> *u, int *ldu, std::complex<double> *vt,
                   int *ldvt, std::complex<double> *work, int *lwork,
                   double *rwork, int *info);
void (*dgesdd_ptr)(char *jobz, int *m, int *n, double *a, int *lda, double *s,
                   double *u, int *ldu, double *vt, int *ldvt, double *work,
                   int *lwork, int *iwork, int *info);
void (*zgesdd_ptr)(char *jobz, int *m, int *n, std::complex<double> *a,
                   int *lda, double *s, std::complex<double> *u, int *ldu,
                   std::complex<double> *vt, int *ldvt,
                   std::complex<double> *work, int *lwork, double *rwork,
                   int *iwork, int *info);

template <class f>
static void load_wrapper(py::dict &__pyx_capi__, const char *name,
                         f *&pointer) {
  py::capsule wrapper = __pyx_capi__[name];
#if 1
  // This copes with a bug in pybind11, which uses
  // static_cast to cast a void* to the pointer.
  pointer = reinterpret_cast<f *>(wrapper.get_pointer<void>());
#else
  pointer = wrapper.get_pointer<f>();
#endif
}

void load_scipy_wrappers() {
  {
    auto cython_blas = py::module_::import("scipy.linalg.cython_blas");
    py::dict __pyx_capi__ = cython_blas.attr("__pyx_capi__");
    load_wrapper(__pyx_capi__, "ddot", ddot_ptr);
    load_wrapper(__pyx_capi__, "zdotc", zdotc_ptr);
    load_wrapper(__pyx_capi__, "dgemm", dgemm_ptr);
    load_wrapper(__pyx_capi__, "zgemm", zgemm_ptr);
  }
  {
    auto cython_lapack = py::module_::import("scipy.linalg.cython_lapack");
    py::dict __pyx_capi__ = cython_lapack.attr("__pyx_capi__");
    load_wrapper(__pyx_capi__, "dgesvd", dgesvd_ptr);
    load_wrapper(__pyx_capi__, "zgesvd", zgesvd_ptr);
    load_wrapper(__pyx_capi__, "dgesdd", dgesdd_ptr);
    load_wrapper(__pyx_capi__, "zgesdd", zgesdd_ptr);
  }
}
} // namespace seemps
