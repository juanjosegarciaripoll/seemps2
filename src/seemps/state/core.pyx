import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

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

cdef void _rescale(cnp.float64_t *data, cnp.float64_t factor, Py_ssize_t N) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(N):
        data[i] /= factor

cdef void _normalize(cnp.float64_t *data, Py_ssize_t N) noexcept nogil:
    _rescale(data, _norm(data, N), N)

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
        _rescale(s_start, sqrt(total - max_error), final_size)
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

cdef cnp.ndarray _as_matrix(cnp.ndarray A, Py_ssize_t rows, Py_ssize_t cols):
    cdef cnp.npy_intp *dims_data = [rows, cols]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 2)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

def _contract_last_and_first(A, B) -> cnp.ndarray:
    """Contract last index of `A` and first from `B`"""
    if (cnp.PyArray_Check(A) == 0 or cnp.PyArray_Check(B) == 0):
        raise ValueError("_contract_last_and_first expects tensors")

    cdef:
        cnp.ndarray Aarray = <cnp.ndarray>A
        cnp.ndarray Barray = <cnp.ndarray>B
        int ndimA = cnp.PyArray_NDIM(Aarray)
        Py_ssize_t Alast = cnp.PyArray_DIM(Aarray, ndimA - 1)
        Py_ssize_t Bfirst = cnp.PyArray_DIM(Barray, 0)
    #
    # By reshaping the two tensors to matrices, we ensure Numpy
    # will always use the CBLAS path (provided the tensors have
    # the same type, of course)
    #
    return <cnp.ndarray>cnp.PyArray_Reshape(
        <cnp.ndarray>cnp.PyArray_MatrixProduct(
            _as_matrix(Aarray, cnp.PyArray_SIZE(Aarray) / Alast, Alast),
            _as_matrix(Barray, Bfirst, cnp.PyArray_SIZE(Barray) / Bfirst)
        ),
        A.shape[:-1] + B.shape[1:])


cdef _matmul = np.matmul

cdef cnp.ndarray _as_3tensor(cnp.ndarray A, Py_ssize_t i, Py_ssize_t j, Py_ssize_t k):
    cdef cnp.npy_intp *dims_data = [i, j, k]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 3)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

cdef cnp.ndarray _as_4tensor(cnp.ndarray A, Py_ssize_t i, Py_ssize_t j,
                             Py_ssize_t k, Py_ssize_t l):
    cdef cnp.npy_intp *dims_data = [i, j, k, l]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 4)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

def _contract_nrjl_ijk_klm(U, A, B) -> cnp.ndarray:
    #
    # Assuming U[n*r,j*l], A[i,j,k] and B[k,l,m]
    # Implements np.einsum('ijk,klm,nrjl -> inrm', A, B, U)
    # See tests.test_contractions for other implementations and timing
    #
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_Check(B) == 0 or
        cnp.PyArray_Check(U) == 0 or
        cnp.PyArray_NDIM(<cnp.ndarray>A) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>B) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>U) != 2):
        raise ValueError("Invalid arguments to _contract_nrjl_ijk_klm")
    cdef:
        # a, d, b = A.shape[0]
        # b, e, c = B.shape[0]
        Py_ssize_t a = cnp.PyArray_DIM(<cnp.ndarray>A, 0)
        Py_ssize_t d = cnp.PyArray_DIM(<cnp.ndarray>A, 1)
        Py_ssize_t b = cnp.PyArray_DIM(<cnp.ndarray>A, 2)
        Py_ssize_t e = cnp.PyArray_DIM(<cnp.ndarray>B, 1)
        Py_ssize_t c = cnp.PyArray_DIM(<cnp.ndarray>B, 2)
    return _as_4tensor(
        _matmul(
            U,
            _as_3tensor(
                <cnp.ndarray>cnp.PyArray_MatrixProduct(
                    _as_matrix(<cnp.ndarray>A, a*d, b),
                    _as_matrix(<cnp.ndarray>B, b, e*c)),
                a, d*e, c)
        ),
        a, d, e, c)
