#pragma once
#include <vector>
#include <algorithm>
#include "core.h"
#include "tensors.h"
#include "strategy.h"
#include "tools.h"

namespace seemps {

template <int rank> class TensorArray {

  static void check_array(py::object A) {
    if (!is_array(A)) {
      throw std::invalid_argument("TensorArray did not get a tensor");
    }
#if 0
    if (array_ndim(A) != rank) {
      throw std::invalid_argument(
          "TensorArray passed tensor of incorrect rank");
    }
#endif
  }

protected:
  std::vector<py::object> data_;

public:
  TensorArray(const TensorArray<rank> &other) : data_(other.data_) {}

  TensorArray(const py::list &data) : data_(data.size()) {
    std::copy(py::begin(data), py::end(data), this->begin());
  }

  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }

  py::object data() const {
    py::list output(size());
    std::copy(data_.begin(), data_.end(), py::begin(output));
    return output;
  }

  void set_data(py::list new_data) {
    auto L = new_data.size();
    data_.resize(L);
    std::copy(py::begin(new_data), py::end(new_data), data_.begin());
  }

  py::object getitem(int k) const {
    auto L = len();
    if (k < 0) {
      k = L + k;
    }
    if (k >= 0 && k < L) {
      return data_[k];
    } else {
      throw std::out_of_range("Wrong index into TensorArray");
    }
  }

  py::object operator[](int k) const { return data_[k]; }

  void setitem(int k, py::object A) {
    check_array(A);
    auto L = len();
    if (k < 0) {
      k = L + k;
    }
    if (k < 0 || k >= L) {
      throw std::out_of_range("Wrong index into TensorArray");
    }
    auto type = array_type(A);
    switch (type) {
    case NPY_DOUBLE:
    case NPY_COMPLEX128:
      data_[k] = A;
      break;
    case NPY_COMPLEX64:
      data_[k] = array_cast(A, NPY_COMPLEX128);
      break;
    default:
      data_[k] = array_cast(A, NPY_DOUBLE);
    }
  }

  py::object __getitem__(py::object site) const {
    auto object = site.ptr();
    if (object == NULL) {
      //
    } else if (PyLong_Check(object)) {
      return getitem(PyLong_AsLong(object));
    } else if (PySlice_Check(object)) {
      Py_ssize_t length = len(), start, stop, step, slicelength;
      py::slice slice = site;
      auto ok = PySlice_GetIndicesEx(site.ptr(), length, &start, &stop, &step,
                                     &slicelength);
      if (ok < 0) {
        throw std::out_of_range("Invalide slize into TensorArray");
      }
      py::list output(slicelength);
      for (Py_ssize_t i = 0; i < slicelength; ++i) {
        output[i] = data_[start];
        start += step;
      }
      return output;
    }
    throw std::invalid_argument("Invalid index into TensorArray");
  }

  py::object __setitem__(py::object site, py::object A) {
    auto object = site.ptr();
    if (object != NULL) {
      if (PyLong_Check(object)) {
        setitem(PyLong_AsLong(object), A);
        return py::none();
      } else if (PySlice_Check(object)) {
        size_t length = data_.size(), start, stop, step, slicelength;
        py::slice slice = site;
        py::sequence new_data = A;
        slice.compute(length, &start, &stop, &step, &slicelength);
        for (size_t i = 0; start < stop; ++i) {
          setitem(start, new_data[i]);
          start += step;
        }
        return py::none();
      }
    }
    throw std::invalid_argument("Invalid index into TensorArray");
  }

  py::object __iter__() const { return py::iter(data()); }

  size_t len() const { return data_.size(); }
  size_t size() const { return data_.size(); }
};

using TensorArray3 = TensorArray<3>;
using TensorArray4 = TensorArray<4>;

using Weight = py::object;

py::object _begin_environment(int D = 1);
py::object _update_left_environment(py::object A, py::object B, py::object rho);
py::object _update_right_environment(py::object A, py::object B,
                                     py::object rho);
Weight _end_environment(py::object rho);
Weight _join_environments(py::object rhoL, py::object rhoR);

Weight scprod(const TensorArray3 &A, const TensorArray3 &B);
double abs(const Weight &);

py::object schmidt_weights(py::object A);

std::tuple<int, double> _update_canonical_right(TensorArray3 &state,
                                                py::object A, int site,
                                                const Strategy &truncation,
                                                bool overwrite = false);
std::tuple<int, double> _update_canonical_left(TensorArray3 &state,
                                               py::object A, int site,
                                               const Strategy &truncation,
                                               bool overwrite = false);
double _canonicalize(TensorArray3 &state, int center,
                     const Strategy &truncation);
std::tuple<py::object, py::object, double>
left_orth_2site(py::object AA, const Strategy &strategy);
std::tuple<py::object, py::object, double>
right_orth_2site(py::object AA, const Strategy &strategy);
double _update_canonical_2site_left(TensorArray3 &state, py::object A, int site,
                                    const Strategy &strategy);
double _update_canonical_2site_right(TensorArray3 &state, py::object A,
                                     int site, const Strategy &strategy);

class MPS : public TensorArray3 {
  double error_;

public:
  MPS(const py::list &data, double error) : TensorArray3(data), error_{error} {}
  MPS(const MPS &data, double error) : TensorArray3(data), error_{error} {}
  MPS(const MPS &) = default;
  MPS(MPS &&) = default;
  MPS &operator=(const MPS &) = default;
  MPS &operator=(MPS &&) = default;
  ~MPS() = default;

  MPS copy() const { return MPS(*this); }
  MPS as_mps() const { return *this; }

  py::int_ dimension() const {
    return std::accumulate(
        begin(), end(), py::int_(1),
        [](py::int_ total_dimension, const py::object &tensor) {
          return total_dimension * py::int_(array_dim(tensor, 1));
        });
  }

  py::list physical_dimensions() const {
    py::list output(size());
    std::transform(
        begin(), end(), py::begin(output),
        [](const py::object &a) -> auto { return py::int_(array_dim(a, 1)); });
    return output;
  }

  py::list bond_dimensions() const {
    py::list output(size() + 1);
    auto start = py::begin(output);
    *start = py::int_(array_dim(data_[0], 0));
    std::transform(begin(), end(), ++start, [](const py::object &a) -> auto {
      return py::int_(array_dim(a, 2));
    });
    return output;
  }

  double norm_squared() const { return py::abs(scprod(*this, *this)); }
  double norm() const { return std::sqrt(norm_squared()); }

  MPS zero_state() {
    MPS output = *this;
    std::transform(begin(), end(), output.begin(), zero_like_array);
    return output;
  }

  auto left_environment(int site) const {
    auto rho = _begin_environment();
    for (int i = 0; i < site; ++i) {
      auto A = getitem(i);
      rho = _update_left_environment(A, A, rho);
    }
    return rho;
  }

  auto right_environment(int site) const {
    auto rho = _begin_environment();
    for (int i = size() - 1; i > site; --i) {
      auto A = getitem(i);
      rho = _update_right_environment(A, A, rho);
    }
    return rho;
  }

  auto error() const { return error_; }

  auto update_error(double delta) { return error_ += sqrt(delta); }

  auto set_error(double new_value) { return error_ = new_value; }

  auto conj_in_place() {
    std::transform(begin(), end(), begin(), array_conjugate);
    return *this;
  }

  auto conj() const { return copy().conj_in_place(); }
};

class MPSSum {
  py::list weights_;
  py::list mps_;
  size_t size_;

public:
  auto size() const { return size_; }
  auto weights() const { return weights_; }
  auto states() const { return mps_; }
  auto sum_size() const { return mps_.size(); }

  MPSSum(py::object weights, py::object states, bool check_args) {
    auto L = py::len(weights);
    if (L != py::len(states)) {
      throw std::invalid_argument(
          "Lists of weights does not match list of states in MPSSum");
    }
    if (L == 0) {
      throw std::invalid_argument("MPSSum requires a non-empty list of states");
    }
    if (!check_args) {
      weights_ = weights;
      mps_ = states;
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
    size_ = mps(0).size();
  }

  auto physical_dimensions() const { return mps(0).physical_dimensions(); }

  auto bond_dimensions() const { return mps(0).bond_dimensions(); }

  auto dimension() const { return mps(0).dimension(); }

  MPSSum copy() const {
    return MPSSum(py::copy(weights_), py::copy(mps_), false);
  }

  MPSSum conj() const {
    auto output = copy();
    for (size_t i = 0; i < sum_size(); ++i) {
      output.weights_[i] = py::conj(output.weights_[i]);
      output.mps_[i].cast<MPS &>().conj_in_place();
    }
  }

  double norm_squared() const {
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

  Weight weight(int site) const { return weights_[site]; }
  MPS &mps(int site) const { return mps_[site].cast<MPS &>(); }

  double norm() const { return std::sqrt(norm_squared()); }

private:
  void append(const Weight &weight, const py::object &mps) {
    weights_.append(weight);
    mps_.append(mps);
  }

  void append(const Weight &factor, const MPSSum &states) {
    const auto &weights = states.weights_;
    const auto &mps = states.mps_;
    for (size_t i = 0; i < states.sum_size(); ++i) {
      append(factor * weights[i], mps[i]);
    }
  }
};

class CanonicalMPS : public MPS {
  int center_;
  Strategy strategy_;

  int interpret_center(int center) const {
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

  int interpret_center_object(const py::object &center,
                              int default_value) const {
    if (center.is_none()) {
      return default_value;
    }
    return interpret_center(center.cast<int>());
  }

public:
  static constexpr int no_defined_center = std::numeric_limits<int>::min();

  CanonicalMPS(const py::list &data, py::object center, double error,
               py::object normalize, const Strategy &strategy,
               bool is_canonical)
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

  CanonicalMPS(const MPS &data, py::object center, double error,
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

  CanonicalMPS(const CanonicalMPS &data, py::object center, double error,
               py::object normalize, const Strategy &strategy,
               bool is_canonical)
      : MPS(data), strategy_{strategy} {
    if (!center.is_none()) {
      recenter(center.cast<int>(), strategy);
    }
    if (normalize.cast<bool>() ||
        (normalize.is_none() && strategy.get_normalize_flag())) {
      normalize_in_place();
    }
  }

  CanonicalMPS(const CanonicalMPS &) = default;
  CanonicalMPS(CanonicalMPS &&) = default;
  CanonicalMPS &operator=(const CanonicalMPS &) = default;
  CanonicalMPS &operator=(CanonicalMPS &&) = default;
  ~CanonicalMPS() = default;

  CanonicalMPS copy() const { return CanonicalMPS(*this); }
  CanonicalMPS as_mps() const { return *this; }

  auto center_tensor() const { return getitem(center_); }

  void normalize_in_place() const {
    auto N = norm();
    if (N) {
      auto A = center_tensor();
      A /= py::float_(N);
    }
  }

  CanonicalMPS zero_state() const {
    auto output = copy();
    std::transform(output.begin(), output.end(), output.begin(),
                   zero_like_array);
    return output;
  }

  double norm_squared() const {
    auto A = center_tensor();
    return abs(array_vdot(A, A));
  }

  py::object left_environment(int site) const {
    auto start = std::min(site, center_);
    auto rho = _begin_environment(array_dim(getitem(start), 0));
    while (start < site) {
      auto A = getitem(start);
      rho = _update_left_environment(A, A, rho);
      ++start;
    }
    return rho;
  }

  py::object right_environment(int site) const {
    auto start = std::max(site, center_);
    auto rho = _begin_environment(array_dim(getitem(start), 2));
    while (start > site) {
      auto A = getitem(start);
      rho = _update_right_environment(A, A, rho);
      --start;
    }
    return rho;
  }

  py::object Schmidt_weights(int site = no_defined_center) const {
    site = interpret_center(site);
    return schmidt_weights(
        ((site == center()) ? (*this) : copy().recenter(site, strategy_))
            .center_tensor());
  }

  double entanglement_entropy(int center) const {
    auto s = Schmidt_weights(center);
    return std::accumulate(s.begin(), s.end(), 0.0,
                           [](double entropy, auto schmidt_weight) -> double {
                             auto w = schmidt_weight.cast<double>();
                             return entropy - w * std::log2(w);
                           });
  }

  double Renyi_entropy(int center, double alpha = 2.0) const {
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

  double update_canonical(py::object A, int direction,
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
  double update_2site_right(py::object AA, int site, const Strategy &strategy) {
    auto error_squared =
        _update_canonical_2site_left(*this, AA, site, strategy);
    center_ = site + 1;
    return update_error(error_squared);
  }

  // FIXME!!! left and right mixed
  double update_2site_left(py::object AA, int site, const Strategy &strategy) {
    auto error_squared =
        _update_canonical_2site_right(*this, AA, site, strategy);
    center_ = site;
    return update_error(error_squared);
  }

  const CanonicalMPS &recenter(int new_center) {
    return recenter(new_center, strategy_);
  }

  const CanonicalMPS &recenter(int new_center, const Strategy &strategy) {

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

  int center() const { return center_; }
};

} // namespace seemps
