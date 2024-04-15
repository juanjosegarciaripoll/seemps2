#include "tools.h"

namespace pybind11 {

object conj(const object &w) {
  if (isinstance<std::complex<double>>(w)) {
    std::complex<double> z = w.cast<std::complex<double>>();
    return cast(std::conj(z));
  } else {
    return w;
  }
}

object real(const object &w) {
  if (isinstance<std::complex<double>>(w)) {
    std::complex<double> z = w.cast<std::complex<double>>();
    return cast(z.real());
  } else {
    return w;
  }
}

double abs(const object &w) { return std::abs(w.cast<std::complex<double>>()); }

list copy(const list &l) {
  list output(l.size());
  std::copy(begin(l), end(l), begin(output));
  return output;
}

} // namespace pybind11
