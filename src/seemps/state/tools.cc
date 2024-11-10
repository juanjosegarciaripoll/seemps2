#include "tools.h"

namespace nanobind {

object conj(const object &w) {
  if (isinstance<std::complex<double>>(w)) {
    std::complex<double> z = cast<std::complex<double>>(w);
    return cast(std::conj(z));
  } else {
    return w;
  }
}

object real(const object &w) {
  if (isinstance<std::complex<double>>(w)) {
    std::complex<double> z = cast<std::complex<double>>(w);
    return cast(z.real());
  } else {
    return w;
  }
}

double abs(const object &w) { return std::abs(cast<std::complex<double>>(w)); }

list copy(const list &l) {
  auto output = empty_list(l.size());
  std::copy(begin(l), end(l), begin(output));
  return output;
}

bool is_true(const object &o) { return bool(bool_(o)); }

} // namespace nanobind
