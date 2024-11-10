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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#if defined(__has_feature)
#if __has_feature(address_sanitizer) // for clang
#define __SANITIZE_ADDRESS__         // GCC already sets this
#endif
#endif

#if defined(__SANITIZE_ADDRESS__) && (defined(__clang__) || defined(__GNUC__))
#define ASAN_IGNORE_FUNCTION __attribute__((no_sanitize_address))
#else
#define ASAN_IGNORE_FUNCTION
#endif

namespace seemps {

namespace py = nanobind;

using namespace nanobind::literals;

} // namespace seemps

#include "tools.h"
