#pragma once
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>

/*
 * We implement this solution to Numpy's symbols not being available
 * https://stackoverflow.com/questions/47026900/pyarray-check-gives-segmentation-fault-with-cython-c
 */
#define PY_ARRAY_UNIQUE_SYMBOL seemps_MY_PyArray_API
// this macro must be defined for the translation unit
#ifndef INIT_NUMPY_ARRAY_CPP
#define NO_IMPORT_ARRAY // for usual translation units
#endif
#include <numpy/arrayobject.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace seemps {

namespace py = pybind11;

}
