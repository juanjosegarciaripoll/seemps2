from cython import *
cimport numpy as cnp
from typing import Optional
from ..typing import Weight, Tensor3, Tensor4, Environment, Operator

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
    SIMPLIFICATION_LAST_CODE = 2

cdef class Strategy:
    cdef int method
    cdef double tolerance
    cdef double simplification_tolerance
    cdef int max_bond_dimension
    cdef int max_sweeps
    cdef bint normalize
    cdef int simplify

cdef class TensorArray:
    cdef list _data
    cdef Py_ssize_t _size

cdef class MPS(TensorArray):
    cdef double _error

cdef class CanonicalMPS(MPS):
    cdef Py_ssize_t _center
    cdef Strategy _strategy

cdef class MPSSum:
    cdef list[Weight] _weights
    cdef list[MPS] _states
    cdef Py_ssize_t _size

cdef class MPO(TensorArray):
    cdef Strategy _strategy

cdef class MPOList:
    cdef list[MPO] _mpos
    cdef Strategy _strategy
    cdef Py_ssize_t _size

cdef class MPOSum:
    cdef list _mpos
    cdef list _weights
    cdef Strategy _strategy
    cdef Py_ssize_t _size

cpdef cnp.ndarray begin_environment()
cpdef cnp.ndarray update_left_environment(
    B: Tensor3, A: Tensor3, rho: Environment, op: Optional[Operator]
)
cpdef cnp.ndarray update_right_environment(
    B: Tensor3, A: Tensor3, rho: Environment, op: Optional[Operator]
)
cpdef object end_environment(rho: Environment)
cpdef object join_environments(rhoL: Environment, rhoR: Environment)

cpdef object scprod(MPS bra, MPS ket)

cpdef cnp.ndarray begin_mpo_environment()
cpdef cnp.ndarray update_left_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
)
cpdef cnp.ndarray update_right_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
)
cpdef object end_mpo_environment(rho: MPOEnvironment)
cpdef object join_mpo_environments(left: MPOEnvironment, right: MPOEnvironment)

cpdef cnp.ndarray _contract_nrjl_ijk_klm(U: Unitary, A: Tensor3, B: Tensor3)
cpdef cnp.ndarray _contract_last_and_first(A: NDArray, B: NDArray)
