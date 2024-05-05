#include "mps.h"

namespace seemps {

py::int_ MPS::dimension() const {
  return std::accumulate(
      begin(), end(), py::int_(1),
      [](py::int_ total_dimension, const py::object &tensor) {
        return total_dimension * py::int_(array_dim(tensor, 1));
      });
}

py::list MPS::physical_dimensions() const {
  py::list output(size());
  std::transform(
      begin(), end(), py::begin(output),
      [](const py::object &a) -> auto { return py::int_(array_dim(a, 1)); });
  return output;
}

py::list MPS::bond_dimensions() const {
  py::list output(size() + 1);
  auto start = py::begin(output);
  *start = py::int_(array_dim(data_[0], 0));
  std::transform(begin(), end(), ++start, [](const py::object &a) -> auto {
    return py::int_(array_dim(a, 2));
  });
  return output;
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
