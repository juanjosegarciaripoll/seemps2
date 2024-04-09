#include "core.h"
#include "tensors.h"

namespace seemps {

void (*dgemm_ptr)(char *, char *, int *, int *, int *, double *, double *,
                  int *, double *, int *, double *, double *, int *);
void (*zgemm_ptr)(char *, char *, int *, int *, int *, std::complex<double> *,
                  std::complex<double> *, int *, std::complex<double> *, int *,
                  std::complex<double> *, std::complex<double> *, int *);

template <class f>
static void load_wrapper(py::dict &__pyx_capi__, const char *name,
                         f *&pointer) {
  py::capsule wrapper = __pyx_capi__[name];
  pointer = wrapper.get_pointer<f>();
}

void load_scipy_wrappers() {
  auto cython_blas = py::module_::import("scipy.linalg.cython_blas");
  py::dict __pyx_capi__ = cython_blas.attr("__pyx_capi__");
  load_wrapper(__pyx_capi__, "dgemm", dgemm_ptr);
  load_wrapper(__pyx_capi__, "zgemm", zgemm_ptr);
}

} // namespace seemps
