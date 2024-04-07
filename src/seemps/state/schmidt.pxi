def ortho_right(object A, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_NDIM(A) != 3):
        raise ValueError()
    cdef:
        cnp.ndarray Aarray = _copy_array(<cnp.ndarray>A)
        Py_ssize_t a = cnp.PyArray_DIM(Aarray, 0)
        Py_ssize_t i = cnp.PyArray_DIM(Aarray, 1)
        Py_ssize_t b = cnp.PyArray_DIM(Aarray, 2)
        Py_ssize_t D
        cnp.ndarray U, s, V
    U, s, V, = __svd(_as_2tensor(Aarray, a*i, b))
    s, err = strategy._truncate(s, strategy)
    D = cnp.PyArray_SIZE(s)
    return _as_3tensor(U[:, :D], a, i, D), _as_2tensor(s, D, 1) * V[:D, :], err


def ortho_left(object A, Strategy strategy) -> tuple[np.ndarray, np.ndarray, float]:
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_NDIM(A) != 3):
        raise ValueError()
    cdef:
        cnp.ndarray Aarray = _copy_array(<cnp.ndarray>A)
        Py_ssize_t a = cnp.PyArray_DIM(Aarray, 0)
        Py_ssize_t i = cnp.PyArray_DIM(Aarray, 1)
        Py_ssize_t b = cnp.PyArray_DIM(Aarray, 2)
        Py_ssize_t D
        cnp.ndarray U, s, V
    U, s, V = __svd(_as_2tensor(Aarray, a, i * b).copy())
    s, err = strategy._truncate(s, strategy)
    D = cnp.PyArray_SIZE(s)
    return _as_3tensor(V[:D, :], D, i, b), U[:, :D] * s, err


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
