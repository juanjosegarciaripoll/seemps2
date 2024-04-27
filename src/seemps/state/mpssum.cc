#include "mps.h"

namespace seemps {

MPSSum::MPSSum(py::object weights, py::object states, bool check_args) {
  if (!check_args) {
    weights_ = std::move(weights);
    mps_ = std::move(states);
  } else {
    auto L = py::len(weights);
    if (L != py::len(states)) {
      throw std::invalid_argument(
          "Lists of weights does not match list of states in MPSSum");
    }
    if (L == 0) {
      throw std::invalid_argument("MPSSum requires a non-empty list of states");
    }
    weights_ = py::list();
    mps_ = py::list();
    for (int i = 0; i < L; ++i) {
      auto wi = weights[py::int_(i)];
      auto si = states[py::int_(i)];
      if (py::isinstance<MPSSum>(si)) {
        append(wi, si.cast<MPSSum>());
      } else if (py::isinstance<MPS>(si)) {
        append(wi, si);
      } else {
        throw std::invalid_argument(
            "MPSSum argument did not contain a valid MPS");
      }
    }
  }
  size_ = mps(0).size();
}

MPSSum::MPSSum(const MPS &mps) : weights_(1), mps_(1), size_{mps.size()} {
  weights_[0] = 1.0;
  mps_[0] = py::cast(mps);
}

MPSSum MPSSum::copy() const {
  return MPSSum(py::copy(weights_), py::copy(mps_), false);
}

MPSSum MPSSum::conj() const {
  auto output = copy();
  for (size_t i = 0; i < sum_size(); ++i) {
    output.weights_[i] = py::conj(output.weights_[i]);
    output.mps_[i].cast<MPS &>().conj_in_place();
  }
  return output;
}

double MPSSum::norm_squared() const {
  double output = 0.0;
  for (size_t i = 0; i < sum_size(); ++i) {
    auto wi_conj = std::conj(weight(i).cast<std::complex<double>>());
    const auto &si = mps(i);
    for (size_t j = i; j < sum_size(); ++j) {
      auto wj = weight(j).cast<std::complex<double>>();
      auto sij = scprod(si, mps(j)).cast<std::complex<double>>();
      auto s = (wi_conj * wj * sij).real() * ((i == j) ? 1 : 2);
      output += s;
    }
  }
  return std::abs(output);
}

double MPSSum::error() const {
  double output = 0.0;
  for (size_t i = 0; i < sum_size(); ++i) {
    auto wi = weights_[i].cast<std::complex<double>>();
    auto error = mps_[i].cast<const MPS &>().error();
    output += std::abs(wi) * error;
  }
  return output;
}

Weight MPSSum::weight(int site) const { return weights_[site]; }

MPS &MPSSum::mps(int site) const { return mps_[site].cast<MPS &>(); }

double MPSSum::norm() const { return std::sqrt(norm_squared()); }

void MPSSum::append(const Weight &weight, const py::object &mps) {
  weights_.append(weight);
  mps_.append(mps);
}

void MPSSum::append(const Weight &factor, const MPSSum &states) {
  const auto &weights = states.weights_;
  const auto &mps = states.mps_;
  for (size_t i = 0; i < states.sum_size(); ++i) {
    append(factor * weights[i], mps[i]);
  }
}

} // namespace seemps
