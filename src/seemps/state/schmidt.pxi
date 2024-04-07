from cpython cimport (
    PyList_Check,
    PyList_GET_SIZE,
    PyList_SetItem,
    PyList_SetItem,
    PyList_GET_ITEM,
    PyTuple_GET_ITEM
    )


cdef tuple[int, float] __update_in_canonical_form_right(
    list[Tensor3] state, cnp.ndarray A, Py_ssize_t site, Strategy truncation
):
    cdef:
        Py_ssize_t a, i, b, D
        cnp.ndarray U, s, V
    if site + 1 == PyList_GET_SIZE(state):
        PyList_SetItem(state, site, A)
        return site, 0.0
    A = _copy_array(A)
    a, i, b = PyArray_DIM(A, 0), PyArray_DIM(A, 1), PyArray_DIM(A,2)
    svd = __svd(_as_2tensor(A, a*i, b))
    U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
    V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
    s, err = truncation._truncate(<cnp.ndarray>PyTuple_GET_ITEM(svd, 1), truncation)
    D = cnp.PyArray_SIZE(s)
    state[site] = _as_3tensor(_resize_matrix(U, -1, D), a, i, D)
    site += 1
    # np.einsum("ab,bic->aic", sV, state[site])
    state[site] = __contract_last_and_first(
        _as_2tensor(s, D, 1) * _resize_matrix(V, D, -1), state[site])
    return site, err


def _update_in_canonical_form_right(state, A, site, truncation):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    assert (PyList_Check(state) and
            cnp.PyArray_Check(A) and
            (cnp.PyArray_NDIM(<cnp.ndarray>A) == 3))
    return __update_in_canonical_form_right(state, <cnp.ndarray>A, site, truncation)


cdef tuple[int, float] __update_in_canonical_form_left(
    list[Tensor3] state, cnp.ndarray A, Py_ssize_t site, Strategy truncation
):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    cdef:
        Py_ssize_t a, i, b, D
        cnp.ndarray U, s, V
    if site == 0:
        PyList_SetItem(state, 0, A)
        return site, 0.0
    A = _copy_array(A)
    a, i, b = PyArray_DIM(A, 0), PyArray_DIM(A, 1), PyArray_DIM(A, 2)
    svd = __svd(_as_2tensor(A, a, i * b))
    U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
    V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
    s, err = truncation._truncate(<cnp.ndarray>PyTuple_GET_ITEM(svd, 1), truncation)
    D = cnp.PyArray_SIZE(s)
    state[site] = _as_3tensor(_resize_matrix(V, D, -1), D, i, b)
    site -= 1
    state[site] = __contract_last_and_first(state[site], _resize_matrix(U, -1, D) * s)
    return site, err

def _update_in_canonical_form_left(state, A, site, truncation):
    assert (PyList_Check(state) and
            cnp.PyArray_Check(A) and
            cnp.PyArray_NDIM(<cnp.ndarray>A) == 3)
    return __update_in_canonical_form_left(state, <cnp.ndarray>A, site, truncation)


def _canonicalize(list[Tensor3] state, int center, Strategy truncation) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    # TODO: Revise the cumulative error update. Does it follow update_error()?
    assert PyList_Check(state)
    cdef:
        Py_ssize_t i, L = PyList_GET_SIZE(state)
        double err = 0.0, errk
        cnp.ndarray A
    for i in range(0, center):
        A = state[i]
        _, errk = __update_in_canonical_form_right(state, A, i, truncation)
        err += errk
    for i in range(L - 1, center, -1):
        A = state[i]
        _, errk = __update_in_canonical_form_left(state, A, i, truncation)
        err += errk
    return err


def left_orth_2site(object AA, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'B' is a left-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    assert (cnp.PyArray_Check(AA) and
            cnp.PyArray_NDIM(<cnp.ndarray>AA) == 4)
    cdef:
        cnp.ndarray A = <cnp.ndarray>AA
        Py_ssize_t a = cnp.PyArray_DIM(A, 0)
        Py_ssize_t d1 = cnp.PyArray_DIM(A, 1)
        Py_ssize_t d2 = cnp.PyArray_DIM(A, 2)
        Py_ssize_t b = cnp.PyArray_DIM(A, 3)
        #
        # Split tensor
        svd = __svd(_as_2tensor(A, a*d1, d2*b))
        cnp.ndarray U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
        cnp.ndarray V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
        cnp.ndarray S = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        #
        # Truncate tensor
        new_S_and_err = strategy._truncate(S, strategy)
        cnp.ndarray new_S = <cnp.ndarray>PyTuple_GET_ITEM(new_S_and_err, 0)
        Py_ssize_t D = cnp.PyArray_SIZE(new_S)
    return (
        _as_3tensor(_resize_matrix(U, -1, D), a, d1, D),
        _as_3tensor(_as_2tensor(new_S, D, 1) * _resize_matrix(V, D, -1), D, d2, b),
        <object>PyTuple_GET_ITEM(new_S_and_err, 1),
    )


def right_orth_2site(object AA, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'C' is a right-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    assert (cnp.PyArray_Check(AA) and
            cnp.PyArray_NDIM(<cnp.ndarray>AA) == 4)
    cdef:
        cnp.ndarray A = <cnp.ndarray>AA
        Py_ssize_t a = cnp.PyArray_DIM(A, 0)
        Py_ssize_t d1 = cnp.PyArray_DIM(A, 1)
        Py_ssize_t d2 = cnp.PyArray_DIM(A, 2)
        Py_ssize_t b = cnp.PyArray_DIM(A, 3)
        #
        # Split tensor
        svd = __svd(_as_2tensor(A, a*d1, d2*b)) # (U, S, V)
        cnp.ndarray U = <cnp.ndarray>PyTuple_GET_ITEM(svd, 0)
        cnp.ndarray V = <cnp.ndarray>PyTuple_GET_ITEM(svd, 2)
        cnp.ndarray S = <cnp.ndarray>PyTuple_GET_ITEM(svd, 1)
        #
        # Truncate tensor
        new_S_and_err = strategy._truncate(S, strategy)
        cnp.ndarray new_S = <cnp.ndarray>PyTuple_GET_ITEM(new_S_and_err, 0)
        Py_ssize_t D = cnp.PyArray_SIZE(new_S)
    return (
        _as_3tensor(_resize_matrix(U, -1, D) * new_S, a, d1, D),
        _as_3tensor(_resize_matrix(V, D, -1), D, d2, b),
        <object>PyTuple_GET_ITEM(new_S_and_err, 1)
    )
