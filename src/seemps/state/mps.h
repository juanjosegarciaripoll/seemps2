#pragma once
#include <vector>
#include "core.h"
#include "tensors.h"
#include "strategy.h"

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

public:
  py::list data_;

  TensorArray(py::object data) : data_(std::move(data)) {}

  py::object data() const { return data_; }

  py::object set_data(py::list data) {
    data_ = std::move(data);
    for (size_t i = 0, L = len(); i < L; ++i) {
      check_array(data_[i]);
    }
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

  py::object __iter__() const { return py::iter(data_); }

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

Weight scprod(py::object A, py::object B);

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
} // namespace seemps
