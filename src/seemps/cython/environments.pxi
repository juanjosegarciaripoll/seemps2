cdef _eye = np.eye
cdef _empty_environment = _eye(1)

def _begin_environment(int D = 1) -> cnp.ndarray:
    """Initiate the computation of a left environment from two MPS."""
    if D == 1:
        return _empty_environment
    return _eye(D)

cdef cnp.ndarray __update_left_environment(object B, object A, object rho):
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_Check(B) == 0 or
        cnp.PyArray_Check(rho) == 0 or
        cnp.PyArray_NDIM(<cnp.ndarray>A) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>B) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>rho) != 2):
        raise ValueError("Invalid arguments to _update_left_environment")
    cdef:
        # i, j, k = A.shape[0]
        # l, j, n = B.shape[0]
        Py_ssize_t i = cnp.PyArray_DIM(<cnp.ndarray>A, 0)
        Py_ssize_t j = cnp.PyArray_DIM(<cnp.ndarray>A, 1)
        Py_ssize_t k = cnp.PyArray_DIM(<cnp.ndarray>A, 2)
        Py_ssize_t l = cnp.PyArray_DIM(<cnp.ndarray>B, 0)
        Py_ssize_t n = cnp.PyArray_DIM(<cnp.ndarray>B, 2)
    # np.einsum("li,ijk->ljk")
    return __gemm(_as_2tensor(<cnp.ndarray>B, l*j, n), GEMM_ADJOINT,
                  _as_2tensor(__gemm(<cnp.ndarray>rho, GEMM_NORMAL,
                                     _as_2tensor(<cnp.ndarray>A, i, j *k),
                                     GEMM_NORMAL),
                              l*j, k),
                  GEMM_NORMAL)

def _update_left_environment(object B, object A, object rho) -> cnp.ndarray :
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product."""
    return __update_left_environment(B, A, rho)

cdef cnp.ndarray __update_right_environment(object B, object A, object rho):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product."""
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_Check(B) == 0 or
        cnp.PyArray_Check(rho) == 0 or
        cnp.PyArray_NDIM(<cnp.ndarray>A) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>B) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>rho) != 2):
        raise ValueError("Invalid arguments to _update_left_environment")
    cdef:
        # i, j, k = A.shape[0]
        # l, j, n = B.shape[0]
        Py_ssize_t i = cnp.PyArray_DIM(<cnp.ndarray>A, 0)
        Py_ssize_t j = cnp.PyArray_DIM(<cnp.ndarray>A, 1)
        Py_ssize_t k = cnp.PyArray_DIM(<cnp.ndarray>A, 2)
        Py_ssize_t l = cnp.PyArray_DIM(<cnp.ndarray>B, 0)
        Py_ssize_t n = cnp.PyArray_DIM(<cnp.ndarray>B, 2)
    # np.einsum("li,ijk->ljk")
    return __gemm(_as_2tensor(__gemm(_as_2tensor(<cnp.ndarray>A, i * j, k),
                                     GEMM_NORMAL,
                                     <cnp.ndarray>rho, GEMM_NORMAL), i, j * n),
                  GEMM_NORMAL,
                  _as_2tensor(<cnp.ndarray>B, l, j * n), GEMM_ADJOINT)

def _update_right_environment(object B, object A, object rho) ->  cnp.ndarray:
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product."""
    return __update_right_environment(B, A, rho)

cdef __end_environment(cnp.ndarray rho):
    return cnp.PyArray_GETITEM(rho, cnp.PyArray_DATA(rho))

def _end_environment(object rho) -> Weight:
    """Extract the scalar product from the last environment."""
    return __end_environment(rho)

# TODO: Separate formats for left- and right- environments so that we
# can replace this with a simple np.dot(ρL.reshape(-1), ρR.reshape(-1))
# This involves ρR -> ρR.T with respect to current conventions
def _join_environments(rhoL: Environment, rhoR: Environment) -> Weight:
    """Join left and right environments to produce a scalar."""
    if cnp.PyArray_DIM(<cnp.ndarray>rhoL, 0) == 1:
        return _end_environment(rhoL) * _end_environment(rhoR)
    return cnp.PyArray_InnerProduct(cnp.PyArray_Ravel(rhoL, cnp.NPY_CORDER),
                                    cnp.PyArray_Ravel(cnp.PyArray_SwapAxes(rhoR, 0, 1),
                                                      cnp.NPY_CORDER))

cdef object _scprod(object bra, object ket):
    cdef:
        list A = bra._data
        list B = ket._data
        Py_ssize_t i
        Py_ssize_t Lbra = cpython.PyList_GET_SIZE(A)
        Py_ssize_t Lket = cpython.PyList_GET_SIZE(B)
    if Lbra != Lket:
        raise ValueError("Invalid arguments to scprod")
    rho = _empty_environment
    for i in range(Lbra):
        rho = __update_left_environment(<cnp.ndarray>cpython.PyList_GET_ITEM(A, i),
                                        <cnp.ndarray>cpython.PyList_GET_ITEM(B, i), rho)
    return __end_environment(rho)

def scprod(object bra, object ket) -> Weight:
    """Compute the scalar product between matrix product states
    :math:`\\langle\\xi|\\psi\\rangle`.

    Parameters
    ----------
    bra : MPS
        Matrix-product state for the bra :math:`\\xi`
    ket : MPS
        Matrix-product state for the ket :math:`\\psi`

    Returns
    -------
    float | complex
        Scalar product.
    """
    return _scprod(bra, ket)

def vdot(object bra, object ket) -> Weight:
    """Alias for :func:`seemps.state.scprod`"""
    return _scprod(bra, ket)

