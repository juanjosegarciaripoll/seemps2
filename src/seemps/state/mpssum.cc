#include "mps.h"

namespace seemps {

MPSSum::MPSSum(py::object weights, py::object states, bool check_args) {
  if (!check_args) {
    weights_ = py::list(weights);
    mps_ = py::list(states);
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
      auto wi = weights[i];
      auto si = states[i];
      if (py::isinstance<MPSSum>(si)) {
        append(wi, py::cast<MPSSum>(si));
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

MPSSum::MPSSum(const MPS &mps)
    : weights_{py::empty_list(1)}, mps_{py::empty_list(1)}, size_{mps.size()} {
  weights_[0] = 1.0;
  mps_[0] = py::cast(mps);
}

MPSSum MPSSum::copy() const {
  return MPSSum(py::copy(weights_), py::copy(mps_), false);
}

MPSSum MPSSum::deepcopy() const {
  py::list mps_list;
  std::for_each(py::begin(mps_), py::end(mps_), [&](py::object a) -> void {
    py::object mps_copy;
    if (py::isinstance<MPS>(a)) {
      mps_copy = py::cast(py::cast<MPS &>(a).deepcopy());
    } else {
      mps_copy = py::cast(py::cast<CanonicalMPS &>(a).deepcopy());
    }
    mps_list.append(mps_copy);
  });
  return MPSSum(py::copy(weights_), mps_list, false);
}

MPSSum MPSSum::conj() const {
  auto output = copy();
  for (size_t i = 0; i < sum_size(); ++i) {
    output.weights_[i] = py::conj(output.weights_[i]);
    py::cast<MPS &>(output.mps_[i]).conj_in_place();
  }
  return output;
}

double MPSSum::norm_squared() const {
  double output = 0.0;
  for (size_t i = 0; i < sum_size(); ++i) {
    auto wi_conj = std::conj(py::cast<std::complex<double>>(weight(i)));
    const auto &si = mps(i);
    for (size_t j = i; j < sum_size(); ++j) {
      auto wj = py::cast<std::complex<double>>(weight(j));
      auto sij = py::cast<std::complex<double>>(scprod(si, mps(j)));
      auto s = (wi_conj * wj * sij).real() * ((i == j) ? 1 : 2);
      output += s;
    }
  }
  return std::abs(output);
}

double MPSSum::error() const {
  double output = 0.0;
  for (size_t i = 0; i < sum_size(); ++i) {
    auto wi = py::cast<std::complex<double>>(weights_[i]);
    auto error = py::cast<const MPS &>(mps_[i]).error();
    output += std::abs(wi) * error;
  }
  return output;
}

Weight MPSSum::weight(int site) const { return weights_[site]; }

MPS &MPSSum::mps(int site) const { return py::cast<MPS &>(mps_[site]); }

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

static double state_norm_squared(const py::object &state) {
  if (py::isinstance<CanonicalMPS>(state)) {
    return py::cast<const CanonicalMPS &>(state).norm_squared();
  } else {
    return py::cast<const MPS &>(state).norm_squared();
  }
}

double MPSSum::delete_zero_components() {
  double c = 0.0;
  py::list final_weights;
  py::list final_states;
  for (int i = 0; i < sum_size(); ++i) {
    std::complex<double> wi = py::cast<std::complex<double>>(weights_[i]);
    if (wi.real() != 0.0 || wi.imag() != 0) {
      auto wic = std::conj(wi);
      auto statei = mps_[i];
      const auto &si = py::cast<const MPS &>(statei);
      auto ni = (wic * wi).real() * state_norm_squared(statei);
      if (ni) {
        for (int j = 0; j < py::len(final_weights); ++j) {
          auto wj = py::cast<std::complex<double>>(final_weights[j]);
          auto sj = py::cast<const MPS &>(final_states[j]);
          c += 2 * (wic * wj * py::cast<std::complex<double>>(scprod(si, sj)))
                       .real();
        }
        final_weights.append(weights_[i]);
        final_states.append(statei);
        c += ni;
      }
    }
  }
  auto L = final_weights.size();
  if (L == 0) {
    final_weights.append(py::cast(0.0));
    final_states.append(py::cast<const MPS &>(mps_[0]).zero_state());
    weights_ = std::move(final_weights);
    mps_ = std::move(final_states);
    return 0.0;
  }
  if (L < sum_size()) {
    weights_ = std::move(final_weights);
    mps_ = std::move(final_states);
  }
  return std::abs(c);
}

} // namespace seemps
