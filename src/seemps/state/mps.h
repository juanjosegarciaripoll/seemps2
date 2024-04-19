#pragma once
#include <vector>
#include <algorithm>
#include "core.h"
#include "tensors.h"
#include "strategy.h"
#include "tools.h"
#include "tensor_array.h"

namespace seemps {

using Weight = py::object;
using Environment = py::object;

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

  py::int_ dimension() const;

  py::list physical_dimensions() const;

  py::list bond_dimensions() const;

  double norm_squared() const;
  double norm() const;

  MPS zero_state() const;

  Environment left_environment(int site) const;

  Environment right_environment(int site) const;

  auto error() const { return error_; }

  auto update_error(double delta) { return error_ += sqrt(delta); }

  auto set_error(double new_value) { return error_ = new_value; }

  auto conj_in_place() {
    std::transform(begin(), end(), begin(), array_conjugate);
    return *this;
  }

  auto conj() const { return copy().conj_in_place(); }

  py::object times_object(const py::object &weight_or_mps) const;
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

  MPSSum(py::object weights, py::object states, bool check_args);

  auto physical_dimensions() const { return mps(0).physical_dimensions(); }

  auto bond_dimensions() const { return mps(0).bond_dimensions(); }

  auto dimension() const { return mps(0).dimension(); }

  MPSSum copy() const;

  MPSSum conj() const;

  double norm_squared() const;

  Weight weight(int site) const;
  MPS &mps(int site) const;

  double norm() const;

  py::object times_object(const py::object &weight_or_mps) const;

private:
  void append(const Weight &weight, const py::object &mps);

  void append(const Weight &factor, const MPSSum &states);
};

class CanonicalMPS : public MPS {
  int center_;
  Strategy strategy_;

  int interpret_center(int center) const;

  int interpret_center_object(const py::object &center,
                              int default_value) const;

public:
  static constexpr int no_defined_center = std::numeric_limits<int>::min();

  CanonicalMPS(const py::list &data, py::object center, double error,
               py::object normalize, const Strategy &strategy,
               bool is_canonical);

  CanonicalMPS(const MPS &data, py::object center, double error,
               py::object normalize, const Strategy &strategy,
               bool is_canonical);

  CanonicalMPS(const CanonicalMPS &data, py::object center, double error,
               py::object normalize, const Strategy &strategy,
               bool is_canonical);

  CanonicalMPS(const CanonicalMPS &) = default;
  CanonicalMPS(CanonicalMPS &&) = default;
  CanonicalMPS &operator=(const CanonicalMPS &) = default;
  CanonicalMPS &operator=(CanonicalMPS &&) = default;
  ~CanonicalMPS() = default;

  CanonicalMPS copy() const { return CanonicalMPS(*this); }
  CanonicalMPS as_mps() const { return *this; }

  auto center_tensor() const { return getitem(center_); }

  void normalize_in_place() const;

  CanonicalMPS zero_state() const;

  double norm_squared() const;

  Environment left_environment(int site) const;

  Environment right_environment(int site) const;

  py::object Schmidt_weights(int site = no_defined_center) const;

  double entanglement_entropy(int center) const;

  double Renyi_entropy(int center, double alpha = 2.0) const;

  double update_canonical(py::object A, int direction,
                          const Strategy &truncation);

  double update_2site_right(py::object AA, int site, const Strategy &strategy);

  double update_2site_left(py::object AA, int site, const Strategy &strategy);

  const CanonicalMPS &recenter(int new_center);

  const CanonicalMPS &recenter(int new_center, const Strategy &strategy);

  py::object times_object(const py::object &weight_or_mps) const;

  int center() const { return center_; }
};

MPSSum operator+(const MPS &a, const MPS &b);
MPSSum operator+(const MPS &a, const MPSSum &b);
MPSSum operator+(const MPSSum &a, const MPSSum &b);
MPSSum operator+(const MPSSum &a, const MPS &b);

MPSSum operator-(const MPS &a, const MPS &b);
MPSSum operator-(const MPSSum &a, const MPS &b);
MPSSum operator-(const MPSSum &a, const MPSSum &b);
MPSSum operator-(const MPS &a, const MPSSum &b);

MPS operator*(const MPS &a, const MPS &b);
inline MPS operator*(int a, const MPS &b) { return double(a) * b; }
MPS operator*(double a, const MPS &b);
MPS operator*(std::complex<double> a, const MPS &b);

inline MPSSum operator*(int a, const MPSSum &b) { return double(a) * b; }
MPSSum operator*(double a, const MPSSum &b);
MPSSum operator*(std::complex<double> a, const MPSSum &b);

inline MPS operator*(const MPS &a, int b) { return double(b) * a; }
inline MPS operator*(const MPS &a, double b) { return b * a; }
inline MPS operator*(const MPS &a, std::complex<double> b) { return b * a; }

CanonicalMPS operator*(double a, const CanonicalMPS &b);
CanonicalMPS operator*(std::complex<double> a, const CanonicalMPS &b);
inline CanonicalMPS operator*(const CanonicalMPS &a, int b) {
  return double(b) * a;
}
inline CanonicalMPS operator*(const CanonicalMPS &a, double b) { return b * a; }
inline CanonicalMPS operator*(const CanonicalMPS &a, std::complex<double> b) {
  return b * a;
}

inline MPSSum operator*(const MPSSum &a, int b) { return double(b) * a; }
inline MPSSum operator*(const MPSSum &a, double b) { return b * a; }
inline MPSSum operator*(const MPSSum &a, std::complex<double> b) {
  return b * a;
}

} // namespace seemps
