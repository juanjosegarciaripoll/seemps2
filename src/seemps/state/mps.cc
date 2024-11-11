#include <numeric>
#include "mps.h"

namespace seemps {

MPS MPS::deepcopy() const {
  auto output = copy();
  std::transform(output.begin(), output.end(), output.begin(), array_copy);
  return output;
}

int MPS::dimension() const {
  return std::accumulate(begin(), end(), int(1),
                         [](int total_dimension, const py::object &tensor) {
                           return total_dimension * int(array_dim(tensor, 1));
                         });
}

py::list MPS::physical_dimensions() const {
  py::list output;
  for (auto a : *this) {
    output.append(array_dim(a, 1));
  }
  return output;
}

py::list MPS::bond_dimensions() const {
  auto output = py::make_list(1);
  for (auto a : *this) {
    output.append(array_dim(a, 2));
  }
  return output;
}

int MPS::max_bond_dimension() const {
  return std::accumulate(begin(), end(), int(1),
                         [](int i, const py::object &a) -> auto {
                           return std::max<int>(i, array_dim(a, 1));
                         });
}

double MPS::norm_squared() const { return py::abs(scprod(*this, *this)); }
double MPS::norm() const { return std::sqrt(norm_squared()); }

void MPS::fill_with_zeros() {
  auto zero_tensor = [](const py::object &A) -> py::object {
    auto d = array_dim(A, 1);
    array_dims_t dims{1, d, 1};
    return zero_array(dims);
  };

  std::transform(begin(), end(), begin(), zero_tensor);
}

MPS MPS::zero_state() const {
  MPS output = copy();
  output.set_error(0.0);
  output.fill_with_zeros();
  return output;
}

Environment MPS::left_environment(int site) const {
  auto rho = _begin_environment();
  for (int i = 0; i < site; ++i) {
    auto A = getitem(i);
    rho = _update_left_environment(A, A, rho);
  }
  return rho;
}

Environment MPS::right_environment(int site) const {
  auto rho = _begin_environment();
  for (int i = size() - 1; i > site; --i) {
    auto A = getitem(i);
    rho = _update_right_environment(A, A, rho);
  }
  return rho;
}

} // namespace seemps
