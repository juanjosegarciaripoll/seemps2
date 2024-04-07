import numpy as np
cimport numpy as cnp
cimport cpython
from libc.math cimport sqrt
from libc.string cimport memcpy
from ..typing  import Environment, Tensor3
from numpy cimport (
    PyArray_Check,
    PyArray_DIM,
    PyArray_NDIM,
    PyArray_SIZE,
    PyArray_DATA
    )

MAX_BOND_DIMENSION = 0x7fffffff
"""Maximum bond dimension for any MPS."""

class Truncation:
    DO_NOT_TRUNCATE = TRUNCATION_DO_NOT_TRUNCATE
    RELATIVE_SINGULAR_VALUE = TRUNCATION_RELATIVE_SINGULAR_VALUE
    RELATIVE_NORM_SQUARED_ERROR = TRUNCATION_RELATIVE_NORM_SQUARED_ERROR
    ABSOLUTE_SINGULAR_VALUE = TRUNCATION_ABSOLUTE_SINGULAR_VALUE

class Simplification:
    DO_NOT_SIMPLIFY = SIMPLIFICATION_DO_NOT_SIMPLIFY
    CANONICAL_FORM = SIMPLIFICATION_CANONICAL_FORM
    VARIATIONAL = SIMPLIFICATION_VARIATIONAL
    VARIATIONAL_EXACT_GUESS = SIMPLIFICATION_VARIATIONAL_EXACT_GUESS

DEFAULT_TOLERANCE = np.finfo(np.float64).eps

cdef class Strategy:
    def __init__(self,
                 method: int = TRUNCATION_RELATIVE_NORM_SQUARED_ERROR,
                 tolerance: float = DEFAULT_TOLERANCE,
                 simplification_tolerance: float = DEFAULT_TOLERANCE,
                 max_bond_dimension: int = MAX_BOND_DIMENSION,
                 normalize: bool = False,
                 simplify: int = SIMPLIFICATION_VARIATIONAL,
                 max_sweeps: int = 16):
        if tolerance < 0 or tolerance >= 1.0:
            raise AssertionError("Invalid tolerance argument passed to Strategy")
        if tolerance == 0 and method != TRUNCATION_DO_NOT_TRUNCATE:
            method = TRUNCATION_ABSOLUTE_SINGULAR_VALUE
        self.tolerance = tolerance
        self.simplification_tolerance = simplification_tolerance
        if max_bond_dimension <= 0 or max_bond_dimension > MAX_BOND_DIMENSION:
            raise AssertionError("Invalid bond dimension in Strategy")
        else:
            self.max_bond_dimension = max_bond_dimension
        self.normalize = normalize
        if simplify < 0 or simplify > SIMPLIFICATION_LAST_CODE:
            raise AssertionError("Invalid simplify argument passed to Strategy")
        else:
            self.simplify = simplify
        if max_sweeps < 0:
            raise AssertionError("Negative or zero number of sweeps in Strategy")
        self.max_sweeps = max_sweeps
        self.method = method
        if method == TRUNCATION_DO_NOT_TRUNCATE:
            self._truncate = _truncate_do_not_truncate
        elif method == TRUNCATION_RELATIVE_NORM_SQUARED_ERROR:
            self._truncate = _truncate_relative_norm_squared_error
        elif method == TRUNCATION_RELATIVE_SINGULAR_VALUE:
            self._truncate = _truncate_relative_singular_value
        elif method == TRUNCATION_ABSOLUTE_SINGULAR_VALUE:
            self._truncate = _truncate_absolute_singular_value
        else:
            raise AssertionError("Invalid method argument passed to Strategy")

    def replace(self,
                 method: Optional[Truncation] = None,
                 tolerance: Optional[float] = None,
                 simplification_tolerance: Optional[float] = None,
                 max_bond_dimension: Optional[int] = None,
                 normalize: Optional[bool] = None,
                 simplify: Optional[int] = None,
                 max_sweeps: Optional[int] = None):
        return Strategy(method = self.method if method is None else method,
                        tolerance = self.tolerance if tolerance is None else tolerance,
                        simplification_tolerance = self.simplification_tolerance if simplification_tolerance is None else simplification_tolerance,
                        max_bond_dimension = self.max_bond_dimension if max_bond_dimension is None else max_bond_dimension,
                        normalize = self.normalize if normalize is None else normalize,
                        simplify = self.simplify if simplify is None else simplify,
                        max_sweeps = self.max_sweeps if max_sweeps is None else max_sweeps)

    def get_method(self) -> int:
        return self.method

    def get_simplification_method(self) -> int:
        return self.simplify

    def get_tolerance(self) -> float:
        return self.tolerance

    def get_simplification_tolerance(self) -> float:
        return self.simplification_tolerance

    def get_max_bond_dimension(self) -> int:
        return self.max_bond_dimension

    def get_normalize_flag(self) -> bool:
        return self.normalize

    def get_max_sweeps(self) -> int:
        return self.max_sweeps

    def get_simplify_flag(self) -> bool:
        return False if self.simplify == 0 else True

    def __str__(self) -> str:
        if self.method == TRUNCATION_DO_NOT_TRUNCATE:
            method="None"
        elif self.method == TRUNCATION_RELATIVE_SINGULAR_VALUE:
            method="RelativeSVD"
        elif self.method == TRUNCATION_RELATIVE_NORM_SQUARED_ERROR:
            method="RelativeNorm"
        elif self.method == TRUNCATION_ABSOLUTE_SINGULAR_VALUE:
            method="AbsoluteSVD"
        else:
            raise ValueError("Invalid truncation method found in Strategy")
        if self.simplify == SIMPLIFICATION_DO_NOT_SIMPLIFY:
            simplification_method="None"
        elif self.simplify == SIMPLIFICATION_CANONICAL_FORM:
            simplification_method="CanonicalForm"
        elif self.simplify == SIMPLIFICATION_VARIATIONAL:
            simplification_method="Variational"
        elif self.simplify == SIMPLIFICATION_VARIATIONAL_EXACT_GUESS:
            simplification_method="Variational (exact guess)"
        else:
            raise ValueError("Invalid simplification method found in Strategy")
        return f"Strategy(method={method}, tolerance={self.tolerance}," \
               f"max_bond_dimension={self.max_bond_dimension}, normalize={self.normalize}," \
               f"simplification_method={simplification_method}, max_sweeps={self.max_sweeps})"

DEFAULT_STRATEGY = Strategy(method = TRUNCATION_RELATIVE_NORM_SQUARED_ERROR,
                            simplify = SIMPLIFICATION_VARIATIONAL,
                            tolerance = DEFAULT_TOLERANCE,
                            simplification_tolerance = DEFAULT_TOLERANCE,
                            max_bond_dimension = MAX_BOND_DIMENSION,
                            normalize = False)

NO_TRUNCATION = DEFAULT_STRATEGY.replace(method = TRUNCATION_DO_NOT_TRUNCATE,
                                         simplify = SIMPLIFICATION_DO_NOT_SIMPLIFY)

cdef tuple _truncate_do_not_truncate(cnp.ndarray s, Strategy strategy):
    return s, 0.0

cdef cnp.ndarray _make_empty_float64_vector(Py_ssize_t N):
    cdef cnp.npy_intp[1] dims = [N]
    return cnp.PyArray_EMPTY(1, &dims[0], cnp.NPY_FLOAT64, 0)

cdef cnp.ndarray _errors_buffer = _make_empty_float64_vector(1024)

cdef cnp.float64_t _norm(cnp.float64_t *data, Py_ssize_t N) noexcept nogil:
    cdef:
        Py_ssize_t i
        cnp.float64_t n = 0.0
    for i in range(N):
        n += data[i] * data[i]
    return sqrt(n)

cdef void _rescale_if_not_zero(cnp.float64_t *data, cnp.float64_t factor, Py_ssize_t N) noexcept nogil:
    cdef Py_ssize_t i
    if factor:
        for i in range(N):
            data[i] /= factor

cdef void _normalize(cnp.float64_t *data, Py_ssize_t N) noexcept nogil:
    _rescale_if_not_zero(data, _norm(data, N), N)

cdef tuple _truncate_relative_norm_squared_error(cnp.ndarray s, Strategy strategy):
    global _errors_buffer
    cdef:
        Py_ssize_t i, final_size, N = s.size
        double max_error, new_norm, final_error
        double total = 0.0
        cnp.float64_t *errors
        cnp.float64_t *s_start = (<cnp.float64_t*>cnp.PyArray_DATA(s))
        cnp.float64_t *data = &s_start[N-1]
    if cnp.PyArray_SIZE(_errors_buffer) <= N:
        _errors_buffer = _make_empty_float64_vector(2 * N)
    errors = <cnp.float64_t*>cnp.PyArray_DATA(_errors_buffer)
    #
    # Compute the cumulative sum of the reduced density matrix eigen values
    # in reversed order. Thus errors[i] is the error we make when we drop
    # i singular values.
    #
    for i in range(N):
        errors[i] = total
        total += data[0]*data[0]
        data -= 1
    errors[N] = total

    max_error = total * strategy.tolerance
    final_error = 0.0
    for i in range(N):
        if errors[i] >= max_error:
            i -= 1
            break
    final_size = min(N - i, strategy.max_bond_dimension)
    max_error = errors[N - final_size]
    if strategy.normalize:
        _rescale_if_not_zero(s_start, sqrt(total - max_error), final_size)
    if final_size < N:
        s = s[:final_size]
    return s, max_error

cdef tuple _truncate_relative_singular_value(cnp.ndarray s, Strategy strategy):
    cdef:
        cnp.float64_t *data = <cnp.float64_t*>cnp.PyArray_DATA(s)
        double max_error = strategy.tolerance * data[0]
        Py_ssize_t i, N = s.size
        Py_ssize_t final_size = min(N, strategy.max_bond_dimension)
    for i in range(1, final_size):
        if data[i] <= max_error:
            final_size = i
            break
    max_error = 0.0
    for i in range(final_size, N):
        max_error += data[i] * data[i]
    if strategy.normalize:
        _normalize(data, final_size)
    if final_size < N:
        s = s[:final_size]
    return s, max_error

cdef tuple _truncate_absolute_singular_value(cnp.ndarray s, Strategy strategy):
    cdef:
        cnp.float64_t *data = <cnp.float64_t*>cnp.PyArray_DATA(s)
        double max_error = strategy.tolerance
        Py_ssize_t i, N = s.size
        Py_ssize_t final_size = min(N, strategy.max_bond_dimension)
    for i in range(1, final_size):
        if data[i] <= max_error:
            final_size = i
            break
    max_error = 0.0
    for i in range(final_size, N):
        max_error += data[i] * data[i]
    if strategy.normalize:
        _normalize(data, final_size)
    if final_size < N:
        s = s[:final_size]
    return s, max_error

def truncate_vector(s, Strategy strategy):
    if (cnp.PyArray_Check(s) == 0 or
        cnp.PyArray_TYPE(<cnp.ndarray>s) != cnp.NPY_FLOAT64 or
        cnp.PyArray_NDIM(<cnp.ndarray>s) != 1):
        raise ValueError("truncate_vector() requires float vector")
    return strategy._truncate(<cnp.ndarray>s, strategy)

include "gemm.pxi"
include "svd.pxi"
include "contractions.pxi"
include "environments.pxi"
include "schmidt.pxi"
