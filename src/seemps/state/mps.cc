#include "mps.h"

namespace seemps {

template <int n> py::object TensorArray<n>::getitem(int k) const {
  auto L = len();
  if (k < 0) {
    k = L + k;
  }
  if (k >= 0 && k < L) {
    py::print("data[k] with k =", k, "flush"_a = true);
    return data_[k];
  } else {
    throw std::out_of_range("Wrong index into TensorArray");
  }
}

template <int n> py::object TensorArray<n>::setitem(int k, py::object) {
  if (!is_array(A)) {
    throw std::invalid_argument("TensorArray did not get a tensor");
  }
  if (array_ndim(A) != rank) {
    throw std::invalid_argument("TensorArray passed tensor of incorrect rank");
  }
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
  return py::none();
}

template <int n> py::object TensorArray<n>::__getitem__(py::object site) const {
  auto object = site.ptr();
  if (object == NULL) {
    //
  } else if (PyLong_Check(object)) {
    return getitem(PyLong_AsLong(object));
  } else if (PySlice_Check(object)) {
    throw std::invalid_argument("Invalid index into TensorArray");
    size_t length = data_.size(), start, stop, step, slicelength;
    py::slice slice = site;
    slice.compute(length, &start, &stop, &step, &slicelength);
    py::list output(length);
    for (size_t i = 0; start < stop; ++i) {
      output[i] = data_[start];
      start += step;
    }
    return output;
  }
  py::print(site, "flush"_a = true);
  throw std::invalid_argument("Invalid index into TensorArray");
}

template <int n>
py::object TensorArray<n>::__setitem__(py::object site, py::object A) {
  auto object = site.ptr();
  if (object == NULL) {
    //
  } else if (PyLong_Check(object)) {
    return setitem(PyLong_AsLong(object), A);
  } else if (PySlice_Check(object)) {
    size_t length = data_.size(), start, stop, step, slicelength;
    py::slice slice = site;
    py::sequence new_data = A;
    slice.compute(length, &start, &stop, &step, &slicelength);
    for (size_t i = 0; start < stop; ++i) {
      data_[start] = new_data[i];
      start += step;
    }
    return py::none();
  }
  py::print(site, "flush"_a = true);
  throw std::invalid_argument("Invalid index into TensorArray");
}

} // namespace seemps
