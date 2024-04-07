from cython import *
cimport numpy as cnp

cdef enum TruncationCEnum:
    TRUNCATION_DO_NOT_TRUNCATE = 0
    TRUNCATION_RELATIVE_SINGULAR_VALUE = 1
    TRUNCATION_RELATIVE_NORM_SQUARED_ERROR = 2
    TRUNCATION_ABSOLUTE_SINGULAR_VALUE = 3
    TRUNCATION_LAST_CODE = 3

cdef enum SimplificationCEnum:
    SIMPLIFICATION_DO_NOT_SIMPLIFY = 0
    SIMPLIFICATION_CANONICAL_FORM = 1
    SIMPLIFICATION_VARIATIONAL = 2
    SIMPLIFICATION_VARIATIONAL_EXACT_GUESS = 3
    SIMPLIFICATION_LAST_CODE = 3

cdef enum GemmFlags:
    GEMM_NORMAL = 0
    GEMM_TRANSPOSE = 1
    GEMM_ADJOINT = 2

cdef class Strategy:
    cdef int method
    cdef double tolerance
    cdef double simplification_tolerance
    cdef int max_bond_dimension
    cdef int max_sweeps
    cdef bint normalize
    cdef int simplify
    cdef double (*_truncate)(cnp.ndarray s, Strategy)
