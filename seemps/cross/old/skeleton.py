import numpy as np
from scipy.linalg import lu, solve_triangular


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


def skeleton(G, i_physical, i_sweep, maxvol_options, ltr):
    """
    Maxvol algorithm.

    This runs the maxvol algorithm on a fiber of the MPS at site i, represented by a matrix G,
    together with some previously computed 'maxvol-optimal' rows given by the indices i_sweep.
    Chooses between square maxvol (maxvol_sqr) or rectangular maxvol (maxvol_rect) according to the
    maxvol parameters and the shape of the fiber G.
    Returns an updated fiber G, and updated i_sweep and a residual matrix R.

    Parameters
    ----------
    G : np.ndarray
        A matrix representing a fiber of the MPS at a given site i.
    i_physical : np.ndarray
        The physical indices of the MPS at site i.
    i_sweep : np.ndarray
        The 'maxvol-optimal' multi-indices of the MPS at site i.
    maxvol_options : dict
        A dictionary containing options for the maxvol algorithm.
    ltr : bool
        A boolean determining if the sweep is in the forward (ltr True) or backward (ltr False) phase.

    Output
    ------
    G_maxvol : np.ndarray
        The updated MPS fiber.
    i_maxvol : np.ndarray
        The updated MPS multi-indices
    R : np.ndarray
        A residual matrix.
    """

    r1, s, r2 = G.shape

    if ltr:
        G = np.reshape(G, (r1 * s, r2), order="F")
    else:
        G = np.reshape(G, (r1, s * r2), order="F").T

    Q, R = np.linalg.qr(G)
    s_Q, r_Q = Q.shape

    if s_Q <= r_Q:  # Square matrix (s_Q = r_Q) or degenerate (s_Q < r_Q)
        i_maxvol = np.arange(s_Q, dtype=int)
        Q_maxvol = np.eye(s_Q, dtype=float)
    elif maxvol_options.maxvol_rct_maxrank == 0:  # Force square maxvol
        i_maxvol, Q_maxvol = maxvol_sqr(
            Q, maxvol_options.sqr_maxiter, maxvol_options.sqr_tau
        )
    else:
        i_maxvol, Q_maxvol = maxvol_rct(
            Q,
            maxvol_options.sqr_maxiter,
            maxvol_options.sqr_tau,
            maxvol_options.rct_tau,
            maxvol_options.rct_min_rank_change,
            maxvol_options.rct_max_rank_change,
        )

    if ltr:
        G_maxvol = np.reshape(Q_maxvol, (r1, s, -1), order="F")
        R = Q[i_maxvol, :] @ R
        i_physical_kron = np.kron(i_physical, _ones(r1))
        if i_sweep is not None:
            i_sweep_kron = np.kron(_ones(s), i_sweep)
            i_physical_kron = np.hstack((i_sweep_kron, i_physical_kron))
    else:
        G_maxvol = np.reshape(Q_maxvol.T, (-1, s, r2), order="F")
        R = (Q[i_maxvol, :] @ R).T
        i_physical_kron = np.kron(_ones(r2), i_physical)
        if i_sweep is not None:
            i_sweep_kron = np.kron(i_sweep, _ones(s))
            i_physical_kron = np.hstack((i_physical_kron, i_sweep_kron))

    i_maxvol = i_physical_kron[i_maxvol, :]
    return G_maxvol, i_maxvol, R


def maxvol_sqr(A, k=100, e=1.05):
    """
    Square maxvol algorithm.
    Given a 'tall matrix' of size m x n (with more rows m than columns n), finds the n rows that
    form a submatrix with approximately the largest volume (absolute value of determinant).

    Parameters
    ----------
    A : np.ndarray
        A tall matrix of size m x n (m > n) to be optimized.
    TODO: Document rest of parameters

    Output
    ------
    I : np.ndarray
        An array with the indices of the n rows that approximate the submatrix with largest volume.
    B : np.ndarray
        The square submatrix of size n x n of A with approximately largest volume.
    """
    n, r = A.shape
    if n <= r:
        raise ValueError('Input matrix should be "tall"')
    P, L, U = lu(A, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T
    for _ in range(k):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if np.abs(B[i, j]) <= e:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return I, B


def maxvol_rct(A, k=100, e=1.05, tau=1.10, min_r=0, max_r=1):
    """
    Rectangular maxvol algorithm.
    Given a 'tall matrix' of size m x n (with more rows m than columns n), finds the \tilde{n} > n rows
    that form a submatrix with approximately the largest volume (absolute value of determinant).

    Parameters
    ----------
    A : np.ndarray
        A tall matrix of size m x n (m > n) to be optimized.
    TODO: Document rest of parameters

    Output
    ------
    I : np.ndarray
        An array with the indices of the \tilde{n} rows that approximate the submatrix with largest volume.
    B : np.ndarray
        The rectangular submatrix of size \tilde{n} x n of A with approximately largest volume.
    """
    n, r = A.shape
    r_min = r + min_r
    r_max = r + max_r if max_r is not None else n
    r_max = min(r_max, n)
    if r_min < r or r_min > r_max or r_max > n:
        raise ValueError("Invalid minimum/maximum number of added rows")
    I0, B = maxvol_sqr(A, k, e)
    I = np.hstack([I0, np.zeros(r_max - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B, axis=1) ** 2
    for k in range(r, r_max):
        i = np.argmax(F)
        if k >= r_min and F[i] <= tau**2:
            break
        I[k] = i
        S[i] = 0
        v = B.dot(B[i])
        l = 1.0 / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)
    I = I[: B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)
    return I, B
