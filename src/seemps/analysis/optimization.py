import numpy as np

from ..state import MPS, CanonicalMPS, MPSSum
from ..cython import _contract_last_and_first
from ..typing import Vector
from .evaluation import evaluate_mps


def get_search_environments(mps: MPS) -> list[Vector]:
    """
    Computes the right environments of the MPS used to accelerate :func:`binary_search_mps`.
    Can be cached and reutilized for subsequent thresholds.
    """
    n = len(mps)
    R = [np.ones((mps[-1].shape[2], 1))]
    for i in reversed(range(n)):
        R.append(_contract_last_and_first(mps[i][:, 1, :], R[-1]))
    return R[::-1]


def binary_search_mps(
    mps: MPS,
    threshold: float,
    increasing: bool = True,
    search_environments: list[Vector] | None = None,
) -> Vector:
    """
    Performs a binary search for the smallest MPS index whose value crosses the given `threshold`.

    Assumes a monotone input, either increasing or decreasing. For efficiency, the required
    search environments can be precomputed and cached using the :func:`get_search_environments` routine.
    """
    dims = mps.physical_dimensions()
    if not all(d == 2 for d in dims):
        raise ValueError(f"This requires binary physical dimensions (got {dims}).")

    R = search_environments or get_search_environments(mps)
    L = np.ones((1, mps[0].shape[0]))
    bits = []
    for i, core in enumerate(mps):
        y = (L @ core[:, 0, :]) @ R[i + 1]
        cond = (threshold <= y) if increasing else (threshold >= y)
        bit = 0 if cond else 1
        bits.append(bit)
        L = L @ core[:, bit, :]
    return np.array(bits, dtype=int)


def optimize_mps(mps: MPS, num_indices: int = 100, make_canonical: bool = True):
    """
    Returns the minimum and maximum values of a given MPS, together with their indices.
    Performs two full sweeps using `optima_tt`, one for the left-to-right and right-to-left
    directions respectively.

    Parameters
    ----------
    mps : MPS
        The MPS to optimize.
    num_indices : int, default=100
        The maximum amount of indices to retain from each tensor. A larger number increases
        the probability of finding the global maxima, but has a larger cost.
    make_canonical : bool, default=True
        Whether to canonicalize the MPS prior to the search and orthogonalize its tensors.

    Returns
    -------
    (i_1, y_1) : tuple
        A tuple with the index and minimum value in the MPS.
    (i_2, y_2) : tuple
        A tuple with the index and maximum value in the MPS.

    Examples
    --------
    .. code-block:: python

        # Compute the two extrema of a given univariate function.
        # Assume that the function is already loaded.
        mps_function_1d = ...
        (i_min, y_min), (i_max, y_max) = optimize_mps(mps)
    """
    # TODO: Optimize (consider simplifying mps_2 and avoiding the product)
    s = mps.physical_dimensions()[0]
    i_1, y_1 = _optima_tt_sweep(mps, num_indices, make_canonical)
    mps_2 = MPSSum(
        weights=[1, -y_1],
        states=[mps, MPS([np.ones((1, s, 1))] * len(mps))],
        check_args=False,
    ).join()
    mps_2 = mps_2 * mps_2
    i_2, _ = _optima_tt_sweep(mps_2, num_indices, make_canonical)
    y_2 = evaluate_mps(mps, i_2)[0]
    if y_1 < y_2:
        return (i_1, y_1), (i_2, y_2)
    else:
        return (i_2, y_2), (i_1, y_1)


def optima_tt(
    mps: MPS,
    num_indices: int = 100,
    make_canonical: bool = True,
    left_to_right: bool = True,
) -> np.ndarray:
    """
    Returns a set of k indices representing k potentially maximum in modulo values of the MPS.
    Performs a probabilistic search traversing the MPS tensors from left-to-right or right-to-left.
    Source: https://arxiv.org/pdf/2209.14808

    Parameters
    ----------
    mps : MPS
        The MPS to optimize.
    num_indices : int, default=100
        The maximum amount of indices to retain from each tensor and return at the end. A larger number
        increases the probability of finding the global maxima, but has a larger cost.
    make_canonical : bool, default=True
        Whether to canonicalize the MPS prior to the search and orthogonalize its tensors.
    left_to_right: bool, default=True
        The direction of the MPS traversal.

    Returns
    -------
    I : np.ndarray
        An array containing `num_indices` indices whose MPS values are potentially maximum in modulo.
    """

    def choose_indices(Q: np.ndarray) -> np.ndarray:
        """Returns the indices of the k rows or columns of Q that are largest norm-2."""
        axis = 1 if left_to_right else 0
        Q_norm = (Q / np.max(np.abs(Q))) ** 2  # Normalize in [0, 1]
        Q_sum = np.sum(Q_norm, axis=axis)
        I_sort = np.argsort(Q_sum)[::-1]  # Indices from largest to smallest norm-2
        return I_sort[:num_indices]

    if left_to_right:
        if make_canonical:
            mps = CanonicalMPS(mps, center=0)
        r_l, s, r_g = mps[0].shape
        Q = mps[0].reshape(r_l * s, r_g)
        I = np.arange(s).reshape(-1, 1)
        ind = choose_indices(Q)
        Q = Q[ind]
        I = I[ind]
        for site in mps[1:]:
            r_l, s, r_g = site.shape
            G = site.reshape(r_l, s * r_g)
            Q = np.matmul(Q, G).reshape(-1, r_g)
            I_old = np.kron(I, np.ones((s, 1), dtype=int))
            I_cur = np.kron(
                np.ones((I.shape[0], 1), dtype=int), np.arange(s).reshape(-1, 1)
            )
            I = np.hstack((I_old, I_cur))
            ind = choose_indices(Q)
            Q = Q[ind]
            I = I[ind]
    else:
        if make_canonical:
            mps = CanonicalMPS(mps, center=-1)
        r_l, s, r_g = mps[-1].shape
        Q = mps[-1].reshape(r_l, s * r_g)
        I = np.arange(s).reshape(-1, 1)
        ind = choose_indices(Q)
        Q = Q[:, ind]
        I = I[ind]
        for site in mps[:-1][::-1]:
            r_l, s, r_g = site.shape
            G = site.reshape(r_l * s, r_g)
            Q = np.matmul(G, Q).reshape(r_l, -1)
            I_old = np.kron(
                np.arange(s).reshape(-1, 1),
                np.ones((I.shape[0], 1), dtype=int),
            )
            I_cur = np.kron(np.ones((s, 1), dtype=int), I)
            I = np.hstack((I_old, I_cur))
            ind = choose_indices(Q)
            Q = Q[:, ind]
            I = I[ind]
    return I


def _optima_tt_sweep(
    mps: MPS,
    num_indices: int,
    make_canonical: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Performs a full sweep (left-right-left) using `optima_tt` and keeps the
    best value from the two.
    """
    i_max_list = [
        optima_tt(mps, num_indices, make_canonical, True)[0],
        optima_tt(mps, num_indices, make_canonical, False)[0],
    ]
    y_max_list = [evaluate_mps(mps, i)[0] for i in i_max_list]
    idx = np.argmax(np.array([abs(y) for y in y_max_list]))
    return i_max_list[idx], y_max_list[idx]
