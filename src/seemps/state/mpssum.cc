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

MPSSum MPSSum::deepcopy() const {
  py::list mps_list;
  std::for_each(py::begin(mps_), py::end(mps_), [&](py::object a) -> void {
    py::object mps_copy;
    if (py::isinstance<MPS>(a)) {
      mps_copy = py::cast(a.cast<MPS &>().deepcopy());
    } else if (py::isinstance<CanonicalMPS>(a)) {
      mps_copy = py::cast(a.cast<CanonicalMPS &>().deepcopy());
    } else {
      mps_copy = py::copy(a);
    }
    mps_list.append(mps_copy);
  });
  return MPSSum(py::copy(weights_), mps_list, false);
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

static double state_norm_squared(const py::object &state) {
  if (py::isinstance<CanonicalMPS>(state)) {
    return state.cast<const CanonicalMPS &>().norm_squared();
  } else {
    return state.cast<const MPS &>().norm_squared();
  }
}

double MPSSum::delete_zero_components() {
  double c = 0.0;
  py::list final_weights(0);
  py::list final_states(0);
  for (int i = 0; i < sum_size(); ++i) {
    std::complex<double> wi = weights_[i].cast<std::complex<double>>();
    if (wi.real() != 0.0 || wi.imag() != 0) {
      auto wic = std::conj(wi);
      auto statei = mps_[i];
      const auto &si = statei.cast<const MPS &>();
      auto ni = (wic * wi).real() * state_norm_squared(statei);
      if (ni) {
        for (int j = 0; j < final_weights.size(); ++j) {
          auto wj = final_weights[j].cast<std::complex<double>>();
          auto sj = final_states[j].cast<const MPS &>();
          c += 2 *
               (wic * wj * scprod(si, sj).cast<std::complex<double>>()).real();
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
    final_states.append(mps_[0].cast<const MPS &>().zero_state());
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
