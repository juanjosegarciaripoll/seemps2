import numpy as np
from scipy.linalg import lu, solve_triangular  # type: ignore


def maxvol_square(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the optimal row indices and matrix of coefficients for the
    square maxvol decomposition of a given matrix.  This algorithm
    finds a submatrix of a 'tall' matrix with maximal volume
    (determinant of the square submatrix).

    Parameters
    ----------
    matrix : np.ndarray
        A 'tall' matrix (more rows than columns) for which the square
        maxvol decomposition is to be computed.

    Returns
    -------
    I : np.ndarray
        The optimal row indices of the tall matrix. These indices
        correspond to rows that form a square submatrix with maximal
        volume.
    B : np.ndarray
        The matrix of coefficients. This matrix represents the
        coefficients for the linear combination of rows in the
        original matrix that approximates the remaining rows, namely,
        a matrix B such that A ≈ B A[I, :].
    """
    SQUARE_MAXITER = 100
    SQUARE_TOL = 1.05
    n, r = matrix.shape
    if n <= r:
        raise ValueError('Input matrix should be "tall"')
    P, L, U = lu(matrix, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, matrix.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T
    for _ in range(SQUARE_MAXITER):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if abs(B[i, j]) <= SQUARE_TOL:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return I, B


def maxvol_rectangular(
    matrix: np.ndarray, min_rank_change: int, max_rank_change: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the optimal row indices and matrix of coefficients for the
    maxvol algorithm applied to a tall matrix.  This algorithm extends
    the square maxvol algorithm to find a 'rectangular' submatrix with
    more rows than the columns of the original matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A 'tall' matrix (more rows than columns) for which the
        rectangular maxvol decomposition is to be computed.
    min_rank_change : int
        The minimum number of rows to be added to the rank of the square submatrix.
    max_rank_change : int
        The maximum number of rows to be added to the rank of the square submatrix.

    Returns
    -------
    I : np.ndarray
        The optimal row indices of the tall matrix. These indices
        correspond to rows that form a rectangular submatrix with more
        rows than the columns of the original matrix.
    B : np.ndarray
        The matrix of coefficients. This matrix represents the
        coefficients of the linear combination of rows in the original
        matrix that approximates the remaining rows, namely, a matrix
        B such that A ≈ B A[I, :].
    """
    RECTANGULAR_TOL = 1.20
    n, r = matrix.shape
    r_min = r + min_rank_change
    r_max = r + max_rank_change if max_rank_change is not None else n
    r_max = min(r_max, n)
    if r_min < r or r_min > r_max or r_max > n:
        raise ValueError("Invalid minimum/maximum number of added rows")
    I0, B = maxvol_square(matrix)
    I = np.hstack([I0, np.zeros(r_max - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B, axis=1) ** 2
    for k in range(r, r_max):
        i = np.argmax(F)
        if k >= r_min and F[i] <= RECTANGULAR_TOL**2:
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


def maxvol(
    matrix: np.ndarray, min_rank_change: int, max_rank_change: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chooses and applies the appropriate maxvol algorithm (square or
    rectangular) to find a submatrix with maximal volume, and returns
    the indices of these rows along with the matrix of coefficients.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix on which the maxvol algorithm is to be applied.
    min_rank_change : int
        The minimum rank change for the matrix when using rectangular maxvol.
    max_rank_change : int
        The maximum rank change for the matrix when using rectangular maxvol.

    Returns
    -------
    indices : np.ndarray
        An array of indices corresponding to the rows that form a
        submatrix with maximal volume.
    coefficients : np.ndarray
        A matrix of coefficients such that A ≈ B A[I, :].
    """

    n, r = matrix.shape
    max_rank_change = min(max_rank_change, n - r)
    min_rank_change = min(min_rank_change, max_rank_change)
    if n <= r:
        indices = np.arange(n, dtype=int)
        coefficients = np.eye(n)
    elif max_rank_change == 0:
        indices, coefficients = maxvol_square(matrix)
    else:
        indices, coefficients = maxvol_rectangular(
            matrix, min_rank_change, max_rank_change
        )

    return indices, coefficients
