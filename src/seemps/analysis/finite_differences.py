from __future__ import annotations
import numpy as np
from ..operators import MPO
from ..register.transforms import mpo_weighted_shifts


def mpo_combined(n, a, b, c, closed=True, **kwdargs):
    A = np.zeros((3, 2, 2, 3))
    # Internal bond dimension 0 is nothing, 1 is add 1, 2 is subtract 1

    A[0, 0, 0, 0] = 1.0
    A[0, 1, 1, 0] = 1.0
    # Increase
    A[0, 1, 0, 1] = 1.0
    A[1, 0, 1, 1] = 1.0
    # Decrease
    A[2, 1, 0, 2] = 1.0
    A[0, 0, 1, 2] = 1.0

    R = a * A[:, :, :, [0]] + b * A[:, :, :, [1]] + c * A[:, :, :, [2]]
    if closed:
        L = A[[0], :, :, :] + A[[1], :, :, :] + A[[2], :, :, :]
    else:
        L = A[[0], :, :, :]
    return MPO([L] + [A] * (n - 2) + [R], **kwdargs)


def finite_differences_mpo(n, Δx, closed=True, **kwdargs):
    if n == 1:
        raise Exception("finite_differences_mpo() does not work with length 1")
    return (1 / Δx**2) * mpo_combined(n, -2, 1, 1, closed=closed, **kwdargs)


_filtered_differences = {
    # First order derivatives
    (1, 3): ([-1 / 2, 1 / 2], [1, -1]),
    (1, 5): ([-1 / 8, -2 / 8, 2 / 8, 1 / 8], [2, 1, -1, -2]),
    (1, 7): (
        [-1 / 32, -4 / 32, -5 / 32, 5 / 32, 4 / 32, 1 / 32],
        [3, 2, 1, -1, -2, -3],
    ),
    (1, 9): (
        [
            -1 / 128,
            -6 / 128,
            -14 / 128,
            -14 / 128,
            14 / 128,
            14 / 128,
            6 / 128,
            1 / 128,
        ],
        [4, 3, 2, 1, -1, -2, -3, -4],
    ),
    (1, 11): (
        [
            -8 / 512,
            -27 / 512,
            -48 / 512,
            -42 / 512,
            42 / 512,
            48 / 512,
            27 / 512,
            8 / 512,
        ],
        [4, 3, 2, 1, -1, -2, -3, -4],
    ),
    # Second order derivatives
    (2, 3): ([1, -2, 1], [-1, 0, 1]),
    (2, 5): ([1 / 4, -2 / 4, 1 / 4], [-2, 0, 2]),
    (2, 7): (
        [1 / 16, 2 / 16, -1 / 16, -4 / 16, -1 / 16, 2 / 16, 1 / 16],
        [-3, -2, -1, 0, 1, 2, 3],
    ),
    (2, 9): (
        [1 / 64, 4 / 64, 4 / 64, -4 / 64, -10 / 64, -4 / 64, 4 / 64, 4 / 64, 1 / 64],
        [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    ),
}


def smooth_finite_differences_mpo(
    L: int,
    order: int,
    filter: int = 3,
    dx: float = 1.0,
    periodic: bool = False,
    base: int = 2,
) -> MPO:
    """Finite differences operator with noise resilience.
    Create the operator that implements a finite-difference approximation to
    the derivative of given `order` for a function encoded in `L`
    units of dimension `base` (which defaults to 2 for qubits). It assumes
    a uniformly spaced grid with separation `dx`.

    Parameters
    ----------
    L : int
        Number of elements in the quantum register
    order : int
        Order of the derivative (currently 1 or 2)
    filter : int, default = 3
        Size of the finite-difference formula with implicit filtering
    dx : float, default = 1.0
        Spacing of the grid
    periodic : bool, default = False
        Whether the grid assumes periodic boundary conditions
    base : int, default = 2
        Quantization of the tensor train (i.e. dimension of the register units)

    Returns
    -------
    operator : MPO
        Matrix product operator encoding the finite difference formula

    Notes
    -----
    See http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators
    """
    key = (order, filter)
    weights, shifts = _filtered_differences.get(key, (None, None))
    if shifts is None:
        raise ValueError(
            "Unknown finite difference derivative of order {order} with noise filter of size {filter}"
        )
    else:
        return mpo_weighted_shifts(
            L,
            np.asarray(weights) / (dx**order),
            shifts,
            periodic=periodic,
            base=base,
        )
