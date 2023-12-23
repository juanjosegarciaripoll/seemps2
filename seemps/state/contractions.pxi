from numpy import matmul as _matmul
from numpy import tensordot as _tensordot

cpdef cnp.ndarray begin_environment():
    """Initiate the computation of a left environment from two MPS. The bond
    dimension defaults to 1. Other values are used for states in canonical
    form that we know how to open and close."""
    return np.eye(1, dtype=np.float64)

cpdef cnp.ndarray begin_environment_with_D(Py_ssize_t D):
    """Initiate the computation of a left environment from two MPS. The bond
    dimension D defaults to 1. Other values are used for states in canonical
    form that we know how to open and close."""
    return np.eye(D, dtype=np.float64)


cpdef cnp.ndarray update_left_environment(
    B: Tensor3, A: Tensor3, rho: Environment, op: Optional[Operator]
):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if op is not None:
        # A = np.einsum("ji,aib->ajb", operator, A)
        A = _matmul(op, A)
    # np.einsum("ijk,li,ljk->nk", A, rho, B.conj())
    i, j, k = A.shape
    l, j, n = B.shape
    # np.einsum("li,ijk->ljk")
    rho = _matmul(rho, A.reshape(i, j * k))
    # np.einsum("nlj,ljk->nk")
    return _matmul(B.reshape(l * j, n).T.conj(), rho.reshape(l * j, k))


cpdef cnp.ndarray update_right_environment(
    B: Tensor3, A: Tensor3, rho: Environment, op: Optional[Operator]
):
    """Extend the left environment with two new tensors, 'B' and 'A' coming
    from the bra and ket of a scalar product. If an operator is provided, it
    is contracted with the ket."""
    if op is not None:
        # A = np.einsum("ji,aib->ajb", operator, A)
        A = _matmul(op, A)
    # np.einsum("ijk,kn,ljn->il", A, rho, B.conj())
    i, j, k = A.shape
    l, j, n = B.shape
    # np.einsum("ijk,kn->ijn", A, rho)
    rho = _matmul(A.reshape(i * j, k), rho)
    return _matmul(rho.reshape(i, j * n), B.reshape(l, j * n).T.conj())


cpdef object end_environment(rho: Environment):
    """Extract the scalar product from the last environment."""
    return rho[0, 0]


# TODO: Separate formats for left- and right- environments so that we
# can replace this with a simple np.dot(rhoL.reshape(-1), rhoR.reshape(-1))
# This involves rhoR -> rhoR.T with respect to current conventions
cpdef object join_environments(rhoL: Environment, rhoR: Environment):
    """Join left and right environments to produce a scalar."""
    # np.einsum("ij,ji", rhoL, rhoR)
    # return np.trace(np.dot(rhoL, rhoR))
    return np.dot(rhoL.reshape(-1), rhoR.T.reshape(-1))


cpdef object scprod(MPS bra, MPS ket):
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
    rho: Environment = begin_environment()
    # TODO: Verify if the order of Ai and Bi matches being bra and ket
    # Add tests for that
    for Ai, Bi in zip(bra, ket):
        rho = update_left_environment(Ai, Bi, rho, None)
    return end_environment(rho)


cpdef cnp.ndarray begin_mpo_environment():
    return np.ones((1, 1, 1), dtype=np.float64)


cpdef cnp.ndarray update_left_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
):
    # output = opt_einsum.contract("acb,ajd,cjie,bif->def", rho, A, O, B)
    # bif,acb->ifac
    aux = _tensordot(B, rho, (0, 2))
    # ifac,cjie->faje
    aux = _tensordot(aux, O, ([0, 3], [2, 0]))
    # faje,ajd-> def
    aux = _tensordot(aux, A, ((1, 2), (0, 1))).transpose(2, 1, 0)
    return aux


cpdef cnp.ndarray update_right_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
):
    # output = opt_einsum.contract("def,ajd,cjie,bif->acb", rho, A, O, B)
    # ajd,def->ajef
    aux = _tensordot(A, rho, (2, 0))
    # ajef,cjie->afci
    aux = _tensordot(aux, O, ((1, 2), (1, 3)))
    # afci,bif->acb
    aux = _tensordot(aux, B, ((1, 3), (2, 1)))
    return aux


cpdef object end_mpo_environment(rho: MPOEnvironment):
    """Extract the scalar product from the last environment."""
    return rho[0, 0, 0]


cpdef object join_mpo_environments(left: MPOEnvironment, right: MPOEnvironment):
    return np.dot(left.reshape(-1), right.reshape(-1))


cpdef cnp.ndarray _contract_nrjl_ijk_klm(U: Unitary, A: Tensor3, B: Tensor3):
    #
    # Assuming U[n*r,j*l], A[i,j,k] and B[k,l,m]
    # Implements np.einsum('ijk,klm,nrjl -> inrm', A, B, U)
    # See tests.test_contractions for other implementations and timing
    #
    a, d, b = A.shape
    b, e, c = B.shape
    return _matmul(
        U, _matmul(A.reshape(-1, b), B.reshape(b, -1)).reshape(a, -1, c)
    ).reshape(a, d, e, c)


cpdef cnp.ndarray _contract_last_and_first(A: NDArray, B: NDArray):
    """Contract last index of `A` and first from `B`"""
    sA = A.shape
    sB = B.shape
    return _matmul(A, B.reshape(sB[0], -1)).reshape(sA[:-1] + sB[1:])
