from cpython cimport (
    PyList_Check,
    PyList_GET_SIZE,
    PyList_SetItem,
    PyList_SetItem,
    PyList_GET_ITEM,
    PyTuple_GET_ITEM,
    Py_INCREF
    )
from libc.math cimport sqrt

cdef inline void state_set(list state, Py_ssize_t n, cnp.ndarray A) noexcept:
     Py_INCREF(A)
     PyList_SetItem(state, n, A)

cdef inline cnp.ndarray state_get(list state, Py_ssize_t n) noexcept:
     return <cnp.ndarray>PyList_GET_ITEM(state, n)

cdef (int, double) __update_in_canonical_form_right(
    list[Tensor3] state, object someA, Py_ssize_t site, Strategy truncation
):
    if site + 1 == PyList_GET_SIZE(state):
        state_set(state, site, someA)
        return site, 0.0
    cdef:
        cnp.ndarray A = _copy_array(someA)
        Py_ssize_t a = PyArray_DIM(A, 0)
        Py_ssize_t i = PyArray_DIM(A, 1)
        Py_ssize_t b = PyArray_DIM(A, 2)
        #
        # Split tensor
        tuple svd = __svd(_as_2tensor(A, a * i, b))
        cnp.ndarray U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
        cnp.ndarray s = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        cnp.ndarray V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
        #
        # Truncate and store in state
        double err = truncation._truncate(s, truncation)
        Py_ssize_t D = PyArray_SIZE(s)
    state_set(state, site, _as_3tensor(_resize_matrix(U, -1, D), a, i, D))
    site += 1
    state_set(state, site, __contract_last_and_first(
        _as_2tensor(s, D, 1) * _resize_matrix(V, D, -1), state_get(state, site)))
    return site, sqrt(err)


def _update_in_canonical_form_right(state, A, site, truncation):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    return __update_in_canonical_form_right(state, <cnp.ndarray>A, site, truncation)


cdef (int, double) __update_in_canonical_form_left(
    list[Tensor3] state, object someA, Py_ssize_t site, Strategy truncation
):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    if site == 0:
        state_set(state, 0, someA)
        return site, 0.0
    cdef:
        cnp.ndarray A = _copy_array(someA)
        Py_ssize_t a = PyArray_DIM(A, 0)
        Py_ssize_t i = PyArray_DIM(A, 1)
        Py_ssize_t b = PyArray_DIM(A, 2)
        tuple svd = __svd(_as_2tensor(A, a, i * b))
        #
        # Split tensor
        cnp.ndarray U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
        cnp.ndarray s = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        cnp.ndarray V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
        #
        # Truncate and store in state
        double err = truncation._truncate(s, truncation)
        Py_ssize_t D = PyArray_SIZE(s)
    state_set(state, site, _as_3tensor(_resize_matrix(V, D, -1), D, i, b))
    site -= 1
    state_set(state, site, __contract_last_and_first(state_get(state, site), _resize_matrix(U, -1, D) * s))
    return site, sqrt(err)

def _update_in_canonical_form_left(state, A, site, truncation):
    return __update_in_canonical_form_left(state, <cnp.ndarray>A, site, truncation)

cdef float __recanonicalize_left(list[Tensor3] state, int oldcenter, int newcenter, Strategy truncation):
    cdef:
        double err = 0.0
    while oldcenter < newcenter:
        err += __update_in_canonical_form_right(state, state_get(state, oldcenter),
                                                oldcenter, truncation)[1]
        oldcenter += 1
    return err

cdef float __recanonicalize_right(list[Tensor3] state, int oldcenter, int newcenter, Strategy truncation):
    cdef:
        double err = 0.0
    while newcenter < oldcenter:
        err += __update_in_canonical_form_left(state, state_get(state, oldcenter),
                                               oldcenter, truncation)[1]
        oldcenter -= 1
    return err

def _recanonicalize(list[Tensor3] state, int oldcenter, int newcenter, Strategy truncation) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    # Invariants:
    # - state is a list
    # - 0 <= oldcenter, newcenter < len(state)
    #
    if oldcenter < newcenter:
        return __recanonicalize_left(state, oldcenter, newcenter, truncation)
    else:
        return __recanonicalize_right(state, oldcenter, newcenter, truncation)


def _canonicalize(list[Tensor3] state, int center, Strategy truncation) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    # Invariants:
    # - state is a list
    # - 0 <= center < len(state)
    #
    return (__recanonicalize_left(state, 0, center, truncation) +
            __recanonicalize_right(state, PyList_GET_SIZE(state)-1, center, truncation))


def _left_orth_2site(object AA, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'B' is a left-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    cdef:
        cnp.ndarray A = <cnp.ndarray>AA
        Py_ssize_t a = PyArray_DIM(A, 0)
        Py_ssize_t d1 = PyArray_DIM(A, 1)
        Py_ssize_t d2 = PyArray_DIM(A, 2)
        Py_ssize_t b = PyArray_DIM(A, 3)
        #
        # Split tensor
        svd = __svd(_as_2tensor(A, a*d1, d2*b))
        cnp.ndarray U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
        cnp.ndarray V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
        cnp.ndarray s = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        #
        # Truncate tensor
        double err = strategy._truncate(s, strategy)
        Py_ssize_t D = PyArray_SIZE(s)
    return (
        _as_3tensor(_resize_matrix(U, -1, D), a, d1, D),
        _as_3tensor(_as_2tensor(s, D, 1) * _resize_matrix(V, D, -1), D, d2, b),
        err,
    )


def _right_orth_2site(object AA, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'C' is a right-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    cdef:
        cnp.ndarray A = <cnp.ndarray>AA
        Py_ssize_t a = PyArray_DIM(A, 0)
        Py_ssize_t d1 = PyArray_DIM(A, 1)
        Py_ssize_t d2 = PyArray_DIM(A, 2)
        Py_ssize_t b = PyArray_DIM(A, 3)
        #
        # Split tensor A into triplet (U, S, V)
        svd = __svd(_as_2tensor(A, a*d1, d2*b))
        cnp.ndarray U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
        cnp.ndarray V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
        cnp.ndarray s = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        #
        # Truncate tensor
        double err = strategy._truncate(s, strategy)
        Py_ssize_t D = PyArray_SIZE(s)
    return (
        _as_3tensor(_resize_matrix(U, -1, D) * s, a, d1, D),
        _as_3tensor(_resize_matrix(V, D, -1), D, d2, b),
        err,
    )
