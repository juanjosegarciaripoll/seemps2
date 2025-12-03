from __future__ import annotations
import numpy as np
from ..mesh import array_affine
from ...typing import Vector


def _get_newton_cotes(nodes: int, cell: Vector) -> Vector:
    """Constructs the Newton-Côtes periodic vector for the given number of nodes and unit cell."""
    l = len(cell)
    if (nodes - l) % (l - 1) != 0:
        raise ValueError("The cell does not fit the number of nodes.")

    k = (nodes - l) // (l - 1)
    q = np.zeros(nodes, dtype=int)
    for i in range(k + 1):
        idx = i * (l - 1)
        q[idx : idx + l] += cell
    return q


def vector_trapezoidal(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the trapezoidal quadrature rule."""
    cell = np.array([1, 1])
    q = _get_newton_cotes(nodes, cell)
    step = (stop - start) / (nodes - 1)
    return (step / 2) * q


def vector_simpson13(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the Simpson 1/3 quadrature rule."""
    cell = np.array([1, 4, 1])
    q = _get_newton_cotes(nodes, cell)
    step = (stop - start) / (nodes - 1)
    return (step / 3) * q


def vector_simpson38(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the Simpson 3/8 quadrature rule."""
    cell = np.array([1, 3, 3, 1])
    q = _get_newton_cotes(nodes, cell)
    step = (stop - start) / (nodes - 1)
    return (3 * step / 8) * q


def vector_boole(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the Boole quadrature rule."""
    cell = np.array([7, 32, 12, 32, 7])
    q = _get_newton_cotes(nodes, cell)
    step = (stop - start) / (nodes - 1)
    return (2 * step / 45) * q


def vector_fifth_order(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the fifth-order quadrature rule."""
    cell = np.array([19, 75, 50, 50, 75, 19])
    q = _get_newton_cotes(nodes, cell)
    step = (stop - start) / (nodes - 1)
    return (5 * step / 288) * q


def vector_best_newton_cotes(start: float, stop: float, nodes: int) -> Vector:
    """Fetches the best Newton-Côtes quadrature rule for the given number o nodes."""
    methods = [
        vector_trapezoidal,
        vector_simpson13,
        vector_simpson38,
        vector_boole,
        vector_fifth_order,
    ]
    for method in methods[::-1]:
        try:
            return method(start, stop, nodes)
        except ValueError:
            continue
    raise Exception("No suitable Newton-Cotes formula found.")


def vector_fejer(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the Fejér quadrature rule."""
    N = nodes
    v = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        v[k] = 2 / (1 - 4 * k**2) * np.exp(1j * k * np.pi / N)
    for k in range(1, N // 2 + 1):
        v[-k] = np.conjugate(v[k])
    if N % 2 == 0:
        v[N // 2] = 0
    h = (stop - start) / 2
    q = np.fft.ifft(v).reshape(-1).real
    return h * q


def vector_clenshaw_curtis(start: float, stop: float, nodes: int) -> Vector:
    """Returns the vector corresponding to the Clenshaw-Curtis quadrature rule."""
    N = nodes
    v = np.zeros(N)
    g = np.zeros(N)
    w0 = 1 / (N**2 - 1 + (N % 2))
    for k in range(N // 2):
        v[k] = 2 / (1 - 4 * k**2)
        g[k] = -w0
    v[N // 2] = (N - 3) / (2 * (N // 2) - 1) - 1
    g[N // 2] = w0 * ((2 - (N % 2)) * N - 1)
    for k in range(1, N // 2 + 1):
        v[-k] = v[k]
        g[-k] = g[k]
    w = np.fft.ifft(v + g).real
    w = np.hstack((w, w[0]))
    return array_affine(w, (-1, 1), (start, stop))
