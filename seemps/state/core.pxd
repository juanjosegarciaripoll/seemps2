from cython import *

cdef enum Truncation:
    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2
    ABSOLUTE_SINGULAR_VALUE = 3
    TRUNCATION_LAST_CODE = 3

cdef enum Simplification:
    DO_NOT_SIMPLIFY = 0
    CANONICAL_FORM = 1
    VARIATIONAL = 2
    SIMPLIFICATION_LAST_CODE = 2

cdef class Strategy:
    cdef int method
    cdef double tolerance
    cdef double simplification_tolerance
    cdef int max_bond_dimension
    cdef int max_sweeps
    cdef bint normalize
    cdef int simplify
