import numpy as np
from ..operators import MPO


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
    return (1 / Δx**2) * mpo_combined(n, -2, 1, 1, closed=closed, **kwdargs)
