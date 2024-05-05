#pragma once
#include <algorithm>
#include "core.h"

namespace seemps {
template <int rank> class TensorArray {

  static void check_array(const py::object &A) {
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
      data_[k] = std::move(A);
      break;
    case NPY_COMPLEX64:
      data_[k] = array_cast(A, NPY_COMPLEX128);
      break;
    default:
      data_[k] = array_cast(A, NPY_DOUBLE);
    }
  }

  py::object __getitem__(const py::object &site) const {
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

  py::object __setitem__(const py::object &site, py::object A) {
    auto object = site.ptr();
    if (object != NULL) {
      if (PyLong_Check(object)) {
        setitem(PyLong_AsLong(object), std::move(A));
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

} // namespace seemps
