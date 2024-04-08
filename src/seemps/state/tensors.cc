#include "tensors.h"

namespace seemps {

/*
 * BLAS Operations
 */

double _norm(const double *data, size_t size) {
  double output = 0.0;
  for (; size; --size, ++data) {
    output += (*data) * (*data);
  }
  return std::sqrt(output);
}

void _normalize(double *data, size_t size, double norm) {
  for (; size; --size, ++data) {
    *data /= norm;
  }
}

void _normalize(double *data, size_t size) {
  return _normalize(data, size, _norm(data, size));
}

} // namespace seemps
