#include <algorithm>
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
  // TODO Fix me!!!
  list output;
  for (auto it = l.begin(), last = l.end(); it != last; ++it) {
    output.append(*it);
  }
  return output;
}

list rescale(const object &factor, const list &b) {
  list c;
  std::for_each(begin(b), end(b),
                [&](object b_i) -> auto { return c.append(factor * b_i); });
  return c;
}

bool is_true(const object &o) { return bool(bool_(o)); }

} // namespace nanobind
