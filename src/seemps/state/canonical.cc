#include "mps.h"

namespace seemps {

int CanonicalMPS::interpret_center(int center) const {
  if (center == no_defined_center) {
    return 0;
  }
  if (center < 0) {
    center = size() + center;
  }
  if (center < 0 || center >= size()) {
    throw std::out_of_range("Invalid center into CanonicalMPS");
  }
  return center;
}

int CanonicalMPS::interpret_center_object(const py::object &center,
                                          int default_value) const {
  if (center.is_none()) {
    return default_value;
  }
  return interpret_center(center.cast<int>());
}

CanonicalMPS::CanonicalMPS(const py::list &data, py::object center,
                           double error, py::object normalize,
                           const Strategy &strategy, bool is_canonical)
    : MPS(data, error), center_{interpret_center_object(center, 0)},
      strategy_{strategy} {
  if (!is_canonical) {
    auto new_error_squared = _canonicalize(*this, center_, strategy_);
    update_error(new_error_squared);
  }
  if (normalize.cast<bool>() ||
      (normalize.is_none() && strategy.get_normalize_flag())) {
    normalize_in_place();
  }
}

CanonicalMPS::CanonicalMPS(const MPS &data, py::object center, double error,
                           py::object normalize, const Strategy &strategy,
                           bool is_canonical)
    : MPS(data), center_{interpret_center_object(center, 0)},
      strategy_{strategy} {
  if (!is_canonical) {
    auto new_error_squared = _canonicalize(*this, center_, strategy_);
    update_error(new_error_squared);
  }
  if (normalize.cast<bool>() ||
      (normalize.is_none() && strategy.get_normalize_flag())) {
    normalize_in_place();
  }
}

CanonicalMPS::CanonicalMPS(const CanonicalMPS &data, py::object center,
                           double error, py::object normalize,
                           const Strategy &strategy, bool is_canonical)
    : MPS(data), strategy_{strategy} {
  if (!center.is_none()) {
    recenter(center.cast<int>(), strategy);
  }
  if (normalize.cast<bool>() ||
      (normalize.is_none() && strategy.get_normalize_flag())) {
    normalize_in_place();
  }
}

CanonicalMPS &CanonicalMPS::normalize_in_place() {
  auto N = norm();
  if (N) {
    auto A = center_tensor();
    A /= py::float_(N);
  }
  return *this;
}

CanonicalMPS CanonicalMPS::zero_state() const {
  auto output = copy();
  std::transform(output.begin(), output.end(), output.begin(), zero_like_array);
  return output;
}

double CanonicalMPS::norm_squared() const {
  auto A = center_tensor();
  return abs(array_vdot(A, A));
}

Environment CanonicalMPS::left_environment(int site) const {
  auto start = std::min(site, center_);
  auto rho = _begin_environment(array_dim(getitem(start), 0));
  while (start < site) {
    auto A = getitem(start);
    rho = _update_left_environment(A, A, rho);
    ++start;
  }
  return rho;
}

Environment CanonicalMPS::right_environment(int site) const {
  auto start = std::max(site, center_);
  auto rho = _begin_environment(array_dim(getitem(start), 2));
  while (start > site) {
    auto A = getitem(start);
    rho = _update_right_environment(A, A, rho);
    --start;
  }
  return rho;
}

py::object CanonicalMPS::Schmidt_weights(int site) const {
  site = interpret_center(site);
  return schmidt_weights(
      ((site == center()) ? (*this) : copy().recenter(site, strategy_))
          .center_tensor());
}

double CanonicalMPS::entanglement_entropy(int center) const {
  auto s = Schmidt_weights(center);
  return std::accumulate(s.begin(), s.end(), 0.0,
                         [](double entropy, auto schmidt_weight) -> double {
                           auto w = schmidt_weight.cast<double>();
                           return entropy - w * std::log2(w);
                         });
}

double CanonicalMPS::Renyi_entropy(int center, double alpha) const {
  auto s = Schmidt_weights(center);
  if (alpha < 0) {
    std::invalid_argument("Invalid Renyi entropy power");
  }
  if (alpha == 0) {
    alpha = 1e-9;
  } else if (alpha == 1) {
    alpha = 1 - 1e-9;
  }
  return std::log(std::accumulate(s.begin(), s.end(), 0.0,
                                  [=](double sum, auto schmidt_weight) {
                                    double w = schmidt_weight.cast<double>();
                                    return sum + std::pow(w, alpha);
                                  })) /
         (1 - alpha);
}

double CanonicalMPS::update_canonical(py::object A, int direction,
                                      const Strategy &truncation) {
  if (direction > 0) {
    auto [new_center, error_squared] =
        _update_canonical_right(*this, A, center_, truncation);
    center_ = new_center;
    return update_error(error_squared);
  } else {
    auto [new_center, error_squared] =
        _update_canonical_left(*this, A, center_, truncation);

    center_ = new_center;
    return update_error(error_squared);
  }
}

// FIXME!!! left and right mixed
double CanonicalMPS::update_2site_right(py::object AA, int site,
                                        const Strategy &strategy) {
  auto error_squared = _update_canonical_2site_left(*this, AA, site, strategy);
  center_ = site + 1;
  return update_error(error_squared);
}

// FIXME!!! left and right mixed
double CanonicalMPS::update_2site_left(py::object AA, int site,
                                       const Strategy &strategy) {
  auto error_squared = _update_canonical_2site_right(*this, AA, site, strategy);
  center_ = site;
  return update_error(error_squared);
}

const CanonicalMPS &CanonicalMPS::recenter(int new_center) {
  return recenter(new_center, strategy_);
}

const CanonicalMPS &CanonicalMPS::recenter(int new_center,
                                           const Strategy &strategy) {

  new_center = interpret_center(new_center);
  auto old_center = center();
  while (center() < new_center) {
    update_canonical(center_tensor(), +1, strategy);
  }
  while (center() > new_center) {
    update_canonical(center_tensor(), -1, strategy);
  }
  return *this;
}

} // namespace seemps
