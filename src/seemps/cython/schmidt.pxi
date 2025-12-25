from cpython cimport (
    PyList_Check,
    PyList_GET_SIZE,
    PyList_SetItem,
    PyList_SetItem,
    PyList_GET_ITEM,
    PyTuple_GET_ITEM,
    Py_INCREF,
    )

cdef void state_set(list state, Py_ssize_t n, cnp.ndarray A) noexcept:
     Py_INCREF(A)
     PyList_SetItem(state, n, A)

cdef inline cnp.ndarray state_get(list state, Py_ssize_t n) noexcept:
     return <cnp.ndarray>PyList_GET_ITEM(state, n)

cdef double __update_in_canonical_form_right(
    list[Tensor3] state, object someA, Py_ssize_t site, Strategy truncation
):
    cdef:
        cnp.ndarray A = _copy_array(<cnp.ndarray>someA)
        Py_ssize_t a = PyArray_DIM(A, 0)
        Py_ssize_t i = PyArray_DIM(A, 1)
        Py_ssize_t b = PyArray_DIM(A, 2)
        #
        # Split tensor
        tuple svd = __svd(_as_2tensor(A, a * i, b))
        #
        # Truncate Schmidt decomposition
        cnp.ndarray s = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        double err = sqrt(truncation._truncate(s, truncation))
        Py_ssize_t D = PyArray_SIZE(s)
        #
        # Build new state tensors
        cnp.ndarray U = _resize_matrix(<cnp.ndarray>PyTuple_GET_ITEM(svd, 0), -1, D)
        cnp.ndarray V = _resize_matrix(<cnp.ndarray>PyTuple_GET_ITEM(svd, 2), D, -1)
    state_set(state, site, _as_3tensor(U, a, i, D))
    site += 1
    state_set(state, site, __contract_last_and_first(_as_2tensor(s, D, 1) * V,
                                                     state_get(state, site)))
    return err


def _update_in_canonical_form_right(state, A, site, truncation) -> tuple[int, float]:
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    #
    # Invariants:
    #  - state is a list
    #  - A is an np.ndarray of rank 3
    #
    if site + 1 == PyList_GET_SIZE(state):
        PyList_SetItem(state, site, A)
        return site, 0.0
    return site+1, __update_in_canonical_form_right(state, A, site, truncation)


cdef double __update_in_canonical_form_left(
    list[Tensor3] state, object someA, Py_ssize_t site, Strategy truncation
):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    cdef:
        cnp.ndarray A = _copy_array(<cnp.ndarray>someA)
        Py_ssize_t a = PyArray_DIM(A, 0)
        Py_ssize_t i = PyArray_DIM(A, 1)
        Py_ssize_t b = PyArray_DIM(A, 2)
        tuple svd = __svd(_as_2tensor(A, a, i * b))
        #
        # Truncate Schmidt decomposition
        cnp.ndarray s = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        double err = sqrt(truncation._truncate(s, truncation))
        Py_ssize_t D = PyArray_SIZE(s)
        #
        # Build new state tensors
        cnp.ndarray U = _resize_matrix(<cnp.ndarray>PyTuple_GET_ITEM(svd, 0), -1, D)
        cnp.ndarray V = _resize_matrix(<cnp.ndarray>PyTuple_GET_ITEM(svd, 2), D, -1)
    state_set(state, site, _as_3tensor(V, D, i, b))
    site -= 1
    state_set(state, site, __contract_last_and_first(state_get(state, site), U * s))
    return err

def _update_in_canonical_form_left(state, A, site, truncation) -> tuple[int, float]:
    #
    # Invariants:
    #  - state is a list
    #  - A is an np.ndarray of rank 3
    #
    if site == 0:
        PyList_SetItem(state, 0, A)
        return 0, 0.0
    return site-1, __update_in_canonical_form_left(state, <cnp.ndarray>A, site, truncation)


def _recanonicalize(list[Tensor3] state, int oldcenter, int newcenter, Strategy truncation) -> float:
    cdef double err = 0.0
    while oldcenter > newcenter:
        err += __update_in_canonical_form_left(state, state_get(state, oldcenter),
                                               oldcenter, truncation)
        oldcenter -= 1
    while oldcenter < newcenter:
        err += __update_in_canonical_form_right(state, state_get(state, oldcenter),
                                                oldcenter, truncation)
        oldcenter += 1
    return err


def _canonicalize(list[Tensor3] state, int center, Strategy truncation) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    # TODO: Revise the cumulative error update. Does it follow update_error()?
    cdef:
        Py_ssize_t i, L = PyList_GET_SIZE(state)
        double err = 0.0
    for i in range(0, center):
        err += __update_in_canonical_form_right(state, state_get(state, i), i, truncation)
    for i in range(L - 1, center, -1):
        err += __update_in_canonical_form_left(state, state_get(state, i), i, truncation)
    return err


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
        sqrt(err),
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
        sqrt(err),
    )
