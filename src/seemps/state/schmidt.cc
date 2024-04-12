#include "core.h"
#include "tensors.h"
#include "strategy.h"
#include "mps.h"

namespace seemps {

std::tuple<int, double>
_update_in_canonical_form_right(py::list state, py::object A, int site,
                                const Strategy &strategy, bool overwrite) {
  if (!is_array(A) || array_ndim(A) != 3) {
    throw std::invalid_argument(
        "Invalid tensor passed to _update_in_canonical_form_right");
  }
  py::object tensor = overwrite ? array_getcontiguous(A) : array_copy(A);
  auto a = array_dim(tensor, 0);
  auto i = array_dim(tensor, 1);
  auto b = array_dim(tensor, 2);

  // Split tensor
  auto [U, s, V] = destructive_svd(as_matrix(tensor, a * i, b));
  double err = destructively_truncate_vector(s, strategy);
  auto D = array_size(s);
  state[site] = array_reshape(matrix_resize(U, -1, D), array_dims_t{a, i, D});
  ++site;
  state[site] = contract_last_and_first(
      as_matrix(s, D, 1) * matrix_resize(V, D, -1), state[site]);
  return {site, err};
}

std::tuple<int, double> _update_in_canonical_form_left(py::list state,
                                                       py::object A, int site,
                                                       const Strategy &strategy,
                                                       bool overwrite) {
  if (!is_array(A) || array_ndim(A) != 3) {
    throw std::invalid_argument(
        "Invalid tensor passed to _update_in_canonical_form_right");
  }
  py::object tensor = overwrite ? array_getcontiguous(A) : array_copy(A);
  auto a = array_dim(tensor, 0);
  auto i = array_dim(tensor, 1);
  auto b = array_dim(tensor, 2);

  // Split tensor
  auto [U, s, V] = destructive_svd(as_matrix(tensor, a, i * b));
  double err = destructively_truncate_vector(s, strategy);
  auto D = array_size(s);
  state[site] = array_reshape(matrix_resize(V, D, -1), array_dims_t{D, i, b});
  --site;
  state[site] =
      contract_last_and_first(state[site], matrix_resize(U, -1, D) * s);
  return {site, err};
}

double _canonicalize(py::list state, int center, const Strategy &strategy) {
  double err = 0.0;
  for (int i = 0; i < center;) {
    auto [site, errk] =
        _update_in_canonical_form_right(state, state[i], i, strategy);
    err += errk;
    i = site;
  }
  for (int i = state.size() - 1; i > center;) {
    auto [site, errk] =
        _update_in_canonical_form_left(state, state[i], i, strategy);
    err += errk;
    i = site;
  }
  return err;
}

std::tuple<py::object, py::object, double>
left_orth_2site(py::object AA, const Strategy &strategy) {
  if (!is_array(AA) || array_ndim(AA) != 4) {
    throw std::invalid_argument("left_orth_2site requires a 4D tensor");
  }
  auto a = array_dim(AA, 0);
  auto d1 = array_dim(AA, 1);
  auto d2 = array_dim(AA, 2);
  auto b = array_dim(AA, 3);
  auto [U, s, V] =
      destructive_svd(array_reshape(AA, array_dims_t{a * d1, d2 * b}));
  auto err = destructively_truncate_vector(s, strategy);
  auto D = array_size(s);
  return {as_3tensor(matrix_resize(U, -1, D), a, d1, D),
          as_3tensor(as_matrix(s, D, 1) * matrix_resize(V, D, -1), D, d2, b),
          err};
}

std::tuple<py::object, py::object, double>
right_orth_2site(py::object AA, const Strategy &strategy) {
  if (!is_array(AA) || array_ndim(AA) != 4) {
    throw std::invalid_argument("left_orth_2site requires a 4D tensor");
  }
  auto a = array_dim(AA, 0);
  auto d1 = array_dim(AA, 1);
  auto d2 = array_dim(AA, 2);
  auto b = array_dim(AA, 3);
  auto [U, s, V] =
      destructive_svd(array_reshape(AA, array_dims_t{a * d1, d2 * b}));
  auto err = destructively_truncate_vector(s, strategy);
  auto D = array_size(s);
  return {as_3tensor(matrix_resize(U, -1, D) * s, a, d1, D),
          as_3tensor(matrix_resize(V, D, -1), D, d2, b), err};
}

/*




def left_orth_2site(AA, strategy: Strategy):
  """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
  that 'B' is a left-isometry, truncating the size 'r' according
  to the given 'strategy'. Tensor 'AA' may be overwritten."""
  α, d1, d2, β = AA.shape
  U, S, V = _destructive_svd(AA.reshape(α * d1, β * d2))
  err = destructively_truncate_vector(S, strategy)
  D = S.size
  return (
      U[:, :D].reshape(α, d1, D),
      (S.reshape(D, 1) * V[:D, :]).reshape(D, d2, β),
      err,
  )


def right_orth_2site(AA, strategy: Strategy):
  """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
  that 'C' is a right-isometry, truncating the size 'r' according
  to the given 'strategy'. Tensor 'AA' may be overwritten."""
  α, d1, d2, β = AA.shape
  U, S, V = _destructive_svd(AA.reshape(α * d1, β * d2))
  err = destructively_truncate_vector(S, strategy)
  D = S.size
  return (U[:, :D] * S).reshape(α, d1, D), V[:D, :].reshape(D, d2, β), err

 */

} // namespace seemps
