#include "tensors.h"
#include "mps.h"

namespace seemps {

template <class elt = double> py::object eye(int D) {
  auto A = zero_matrix(D, D);
  auto base = array_data<elt>(A);
  for (int i = 0; i < D; ++i) {
    base[i + i * D] = static_cast<elt>(1);
  }
  return A;
}

py::object _begin_environment(int D) {
#if 0
  bool initialized = 0;
  static py::object _empty_environment;

  if (D == 1) {
    if (!initialized) {
      _empty_environment = eye(1);
    }
    return _empty_environment;
  }
#endif
  return eye(D);
}

py::object _update_left_environment(py::object B, py::object A,
                                    py::object rho) {
  if (!is_array(A) || !is_array(B) || !is_array(rho) || (array_ndim(B) != 3) ||
      (array_ndim(A) != 3) || (array_ndim(rho) != 2)) {
    throw std::invalid_argument(
        "Invalid or non-matching tensors in _update_left_environment");
  }

  auto i = array_dim(A, 0);
  auto j = array_dim(A, 1);
  auto k = array_dim(A, 2);
  auto l = array_dim(B, 0);
  auto n = array_dim(B, 2);
  return gemm(
      as_matrix(B, l * j, n), GEMM_ADJOINT,
      as_matrix(gemm(rho, GEMM_NORMAL, as_matrix(A, i, j * k), GEMM_NORMAL),
                l * j, k),
      GEMM_NORMAL);
}

py::object _update_right_environment(py::object B, py::object A,
                                     py::object rho) {
  if (!is_array(A) || !is_array(B) || !is_array(rho) || (array_ndim(B) != 3) ||
      (array_ndim(A) != 3) || (array_ndim(rho) != 2)) {
    throw std::invalid_argument(
        "Invalid or non-matching tensors in _update_left_environment");
  }

  auto i = array_dim(A, 0);
  auto j = array_dim(A, 1);
  auto k = array_dim(A, 2);
  auto l = array_dim(B, 0);
  auto n = array_dim(B, 2);
  return gemm(
      as_matrix(gemm(as_matrix(A, i * j, k), GEMM_NORMAL, rho, GEMM_NORMAL), i,
                j * n),
      GEMM_NORMAL, as_matrix(B, l, j * n), GEMM_ADJOINT);
}

Weight _end_environment(py::object rho) {
  auto A = to_array(rho);
  return py::reinterpret_steal<py::object>(
      PyArray_GETITEM(A, static_cast<char *>(PyArray_DATA(A))));
}

/*
 * TODO: Separate formats for left- and right- environments so that we
 * can replace this with a simple np.dot(ρL.reshape(-1), ρR.reshape(-1))
 * This involves ρR -> ρR.T with respect to current conventions
 */
Weight _join_environments(py::object rhoL, py::object rhoR) {
  if (array_dim(rhoL, 0) == 1) {
    return _end_environment(rhoL) * _end_environment(rhoR);
  }
  return py::reinterpret_steal<py::object>(PyArray_InnerProduct(
      PyArray_Ravel(to_array(rhoL), NPY_CORDER),
      PyArray_Ravel(reinterpret_cast<PyArrayObject *>(
                        PyArray_SwapAxes(to_array(rhoR), 0, 1)),
                    NPY_CORDER)));
}

Weight scprod(py::object bra, py::object ket) {
  py::list A = bra.attr("_data");
  py::list B = ket.attr("_data");
  auto Lbra = A.size();
  auto Lket = B.size();
  if (Lbra != Lket) {
    throw std::invalid_argument("Non matching MPS in scprod()");
  }
  auto rho = _begin_environment();
  for (size_t i = 0; i < Lbra; ++i) {
    rho = _update_left_environment(A[i], B[i], rho);
  }
  return _end_environment(rho);
}

} // namespace seemps
