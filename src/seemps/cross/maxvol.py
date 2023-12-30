from typing import Tuple
import numpy as np
from scipy.linalg import lu, solve_triangular  # type: ignore


def maxvol_sqr(
    A: np.ndarray, k: int = 100, e: float = 1.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Square maxvol algorithm.
    Given a 'tall matrix' of size m x n (with more rows m than columns n), finds the n rows that
    form a submatrix with approximately the largest volume (absolute value of determinant).

    Parameters
    ----------
    A : np.ndarray
        A tall matrix of size m x n (m > n) to be optimized.
    k : int
        The maximum iterations allowed for the algorithm to converge.
    e : float
        The sensitivity of the algorithm (e >= 1).

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


def maxvol_rct(
    A: np.ndarray,
    k: int = 100,
    e: float = 1.05,
    tau: float = 1.10,
    min_r: int = 0,
    max_r: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectangular maxvol algorithm.
    Given a 'tall matrix' of size m x n (with more rows m than columns n), finds the \tilde{n} > n rows
    that form a submatrix with approximately the largest volume (absolute value of determinant).

    Parameters
    ----------
    A : np.ndarray
        A tall matrix of size m x n (m > n) to be optimized.
    k : int
        The maximum iterations allowed for the square maxvol algorithm to converge.
    e : float
        The sensitivity of the square maxvol algorithm (e >= 1).
    tau : float
        The sensitivity of the rectangular maxvol algorithm (tau >= 1).
    min_r : int
        The minimum rank increment (added rows) introduced by the rectangular maxvol algorithm.
    max_r : int
        The maximum rank increment (added rows) introduced by the rectangular maxvol algorithm.

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
