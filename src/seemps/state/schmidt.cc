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

} // namespace seemps
