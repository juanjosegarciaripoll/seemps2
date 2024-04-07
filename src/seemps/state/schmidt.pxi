from cpython cimport (
    PyList_Check,
    PyList_GET_SIZE,
    PyList_SetItem,
    PyList_SetItem,
    PyList_GET_ITEM
    )

cdef tuple[np.ndarray, np.ndarray, float] _ortho_right(cnp.ndarray A, Strategy strategy):
    cdef:
        cnp.ndarray Aarray = _copy_array(A)
        Py_ssize_t a = cnp.PyArray_DIM(Aarray, 0)
        Py_ssize_t i = cnp.PyArray_DIM(Aarray, 1)
        Py_ssize_t b = cnp.PyArray_DIM(Aarray, 2)
        Py_ssize_t D
        cnp.ndarray U, s, V
    U, s, V, = __svd(_as_2tensor(Aarray, a*i, b))
    s, err = strategy._truncate(s, strategy)
    D = cnp.PyArray_SIZE(s)
    return _as_3tensor(U[:, :D], a, i, D), _as_2tensor(s, D, 1) * V[:D, :], err


cdef tuple[np.ndarray, np.ndarray, float] _ortho_left(cnp.ndarray A, Strategy strategy):
    cdef:
        cnp.ndarray Aarray = _copy_array(A)
        Py_ssize_t a = cnp.PyArray_DIM(Aarray, 0)
        Py_ssize_t i = cnp.PyArray_DIM(Aarray, 1)
        Py_ssize_t b = cnp.PyArray_DIM(Aarray, 2)
        Py_ssize_t D
        cnp.ndarray U, s, V
    U, s, V = __svd(_as_2tensor(Aarray, a, i * b).copy())
    s, err = strategy._truncate(s, strategy)
    D = cnp.PyArray_SIZE(s)
    return _as_3tensor(V[:D, :], D, i, b), U[:, :D] * s, err


cdef tuple[int, float] __update_in_canonical_form_right(
    list[Tensor3] state, cnp.ndarray A, Py_ssize_t site, Strategy truncation
):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    cdef:
        Py_ssize_t L = PyList_GET_SIZE(state)
    if site + 1 == L:
        PyList_SetItem(state, site, A)
        return site, 0.0
    A, sV, err = _ortho_right(<cnp.ndarray>A, truncation)
    state[site] = A
    site += 1
    # np.einsum("ab,bic->aic", sV, state[site])
    state[site] = __contract_last_and_first(<cnp.ndarray>sV, state[site])
    return site, err


def _update_in_canonical_form_right(state, A, site, truncation):
    if (PyList_Check(state) == 0 or
        cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_NDIM(A) != 3):
        raise ValueError()
    return __update_in_canonical_form_right(state, <cnp.ndarray>A, site, truncation)


cdef tuple[int, float] __update_in_canonical_form_left(
    list[Tensor3] state, cnp.ndarray A, Py_ssize_t site, Strategy truncation
):
    """Insert a tensor in canonical form into the MPS state at the given site.
    Update the neighboring sites in the process."""
    if (PyList_Check(state) == 0 or
        cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_NDIM(A) != 3):
        raise ValueError()
    if site == 0:
        PyList_SetItem(state, 0, A)
        return site, 0.0
    A, Us, err = _ortho_left(<cnp.ndarray>A, truncation)
    state[site] = A
    site -= 1
    # np.einsum("aib,bc->aic", state[site], Us)
    state[site] = __contract_last_and_first(state[site], <cnp.ndarray>Us)
    return site, err

def _update_in_canonical_form_left(state, A, site, truncation):
    if (PyList_Check(state) == 0 or
        cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_NDIM(A) != 3):
        raise ValueError()
    return __update_in_canonical_form_left(state, <cnp.ndarray>A, site, truncation)


def _canonicalize(list[Tensor3] state, int center, Strategy truncation) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    # TODO: Revise the cumulative error update. Does it follow update_error()?
    if (PyList_Check(state) == 0):
        raise ValueError()
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
    if (cnp.PyArray_Check(AA) == 0 or
        cnp.PyArray_NDIM(AA) != 4):
        raise ValueError()
    cdef:
        cnp.ndarray AAarray = <cnp.ndarray>AA
        Py_ssize_t a = cnp.PyArray_DIM(AAarray, 0)
        Py_ssize_t d1 = cnp.PyArray_DIM(AAarray, 1)
        Py_ssize_t d2 = cnp.PyArray_DIM(AAarray, 2)
        Py_ssize_t b = cnp.PyArray_DIM(AAarray, 3)
        cnp.ndarray U, S, V
    U, S, V = __svd(_as_2tensor(AAarray, a*d1, d2*b))
    S, err = strategy._truncate(S, strategy)
    D = cnp.PyArray_SIZE(S)
    return (
        _as_3tensor(U[:, :D], a, d1, D),
        _as_3tensor(_as_2tensor(S, D, 1) * V[:D,:], D, d2, b),
        err,
    )


def right_orth_2site(object AA, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'C' is a right-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    if (cnp.PyArray_Check(AA) == 0 or
        cnp.PyArray_NDIM(AA) != 4):
        raise ValueError()
    cdef:
        cnp.ndarray AAarray = <cnp.ndarray>AA
        Py_ssize_t a = cnp.PyArray_DIM(AAarray, 0)
        Py_ssize_t d1 = cnp.PyArray_DIM(AAarray, 1)
        Py_ssize_t d2 = cnp.PyArray_DIM(AAarray, 2)
        Py_ssize_t b = cnp.PyArray_DIM(AAarray, 3)
        cnp.ndarray U, S, V
    U, S, V = __svd(_as_2tensor(AAarray, a*d1, d2*b))
    S, err = strategy._truncate(S, strategy)
    D = cnp.PyArray_SIZE(S)
    return (
        _as_3tensor(U[:, :D] * S, a, d1, D),
        _as_3tensor(V[:D, :], D, d2, b),
        err
    )
