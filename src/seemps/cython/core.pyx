import numpy as np
cimport numpy as cnp
cimport cpython
from libc.math cimport sqrt
from libc.string cimport memcpy
from ..typing  import Environment, Tensor3, Weight
from numpy cimport (
    PyArray_Check,
    PyArray_DIM,
    PyArray_NDIM,
    PyArray_SIZE,
    PyArray_DATA,
    PyArray_Resize,
    PyArray_DIMS,
    )

__version__ = 'cython-contractions'

include "truncation.pxi"
include "gemm.pxi"
include "svd.pxi"
include "contractions.pxi"
include "environments.pxi"
include "schmidt.pxi"
