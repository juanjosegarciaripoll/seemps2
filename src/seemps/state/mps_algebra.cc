#include <pybind11/complex.h>
#include "mps.h"

namespace seemps {

MPSSum operator+(const MPS &a, const MPS &b) {
  py::list weights(2);
  py::list states(2);
  weights[0] = 1.0;
  weights[1] = 1.0;
  states[0] = a;
  states[1] = b;
  return MPSSum(weights, states, false);
}

MPSSum operator+(const MPSSum &a, const MPSSum &b) {
  return MPSSum(a.weights() + b.weights(), a.states() + b.states(), false);
}

MPSSum operator+(const MPS &a, const MPSSum &b) { return MPSSum(a) + b; }

MPSSum operator+(const MPSSum &a, const MPS &b) {
  auto output = a.copy();
  output.weights().append(1.0);
  output.states().append(b);
  return output;
}

MPSSum operator-(const MPS &a, const MPS &b) {
  py::list weights(2);
  py::list states(2);
  weights[0] = 1.0;
  weights[1] = -1.0;
  states[0] = a;
  states[1] = b;
  return MPSSum(weights, states, false);
}

static py::list rescale(const py::object &a, const py::list &b) {
  py::list c(b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    c[i] = a * b[i];
  }
  return c;
}

MPSSum operator-(const MPSSum &a, const MPSSum &b) {
  return MPSSum(a.weights() + rescale(py::float_(-1.0), b.weights()),
                a.states() + b.states(), true);
}

MPSSum operator-(const MPS &a, const MPSSum &b) { return MPSSum(a) - b; }

MPSSum operator-(const MPSSum &a, const MPS &b) {
  py::list weights = copy(a.weights());
  py::list states = copy(a.states());
  weights.append(-1.0);
  states.append(b);
  return MPSSum(weights, states, false);
}

MPS operator*(double a, const MPS &b) {
  if (a == 0) {
    return b.zero_state();
  } else {
    auto c = b.copy();
    c.setitem(0, py::float_(a) * c[0]);
    return c;
  }
}

MPS operator*(std::complex<double> a, const MPS &b) {
  if (a == std::complex<double>(0.0)) {
    return b.zero_state();
  } else {
    auto c = b.copy();
    c.setitem(0, py::cast(a) * c[0]);
    return c;
  }
}

MPS operator*(const MPS &a, const MPS &b) {
  auto output = b.copy();
  output.set_error(0);
  if (output.size() != a.size()) {
    throw std::invalid_argument("Non-matching MPS found in '*'");
  }
  for (size_t n = 0; n < a.size(); ++n) {
    auto A = a[n];
    auto B = b[n];
    // np.einsum('aib,cid->acibd', A, B)
    auto a = array_dim(A, 0);
    auto i = array_dim(A, 1);
    auto b = array_dim(A, 2);
    auto c = array_dim(B, 0);
    auto j = array_dim(B, 1);
    auto d = array_dim(B, 2);
    if (i != j) {
      throw std::invalid_argument("Non-conformant MPS found in '*'");
    }
    py::object rA = array_reshape(A, array_dims_t{a, 1, i, b, 1});
    py::object rB = array_reshape(B, array_dims_t{1, c, i, 1, d});
    output.setitem(n, as_3tensor(rA * rB, a * c, i, b * d));
  }
  return output;
}

CanonicalMPS operator*(double a, const CanonicalMPS &b) {
  if (a == 0) {
    return b.zero_state();
  } else {
    auto c = b.copy();
    c.setitem(0, py::float_(a) * c[0]);
    return c;
  }
}

CanonicalMPS operator*(std::complex<double> a, const CanonicalMPS &b) {
  if (a == std::complex<double>(0.0)) {
    return b.zero_state();
  } else {
    auto c = b.copy();
    c.setitem(0, py::cast(a) * c[0]);
    return c;
  }
}

MPSSum operator*(double a, const MPSSum &b) {
  // TODO: Add zero_state() outcome
  return MPSSum(rescale(py::float_(a), b.weights()), b.states(), false);
}

MPSSum operator*(std::complex<double> a, const MPSSum &b) {
  // TODO: Add zero_state() outcome
  return MPSSum(rescale(py::cast(a), b.weights()), b.states(), false);
}

py::object MPSSum::times_object(const py::object &weight_or_mps) const {
  if (py::isinstance<py::int_>(weight_or_mps)) {
    return py::cast(*this * weight_or_mps.cast<double>());
  } else if (py::isinstance<py::float_>(weight_or_mps)) {
    return py::cast(*this * weight_or_mps.cast<double>());
  } else if (py::iscomplex(weight_or_mps)) {
    return py::cast(*this * weight_or_mps.cast<std::complex<double>>());
  }
  // TODO: Add MPS, CanonicalMPS and other MPSSum
  throw py::type_error("Invalid argument to product by MPS");
}

} // namespace seemps
