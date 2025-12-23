from __future__ import annotations
import numpy as np
from ..operators import MPO, MPOList, MPOSum
from ..state import Strategy, DEFAULT_STRATEGY


def id_mpo(n_qubits: int, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """Identity MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.

    Returns
    -------
    MPO
        Identity operator MPO.
    """
    B = np.zeros((1, 2, 2, 1))
    B[0, 0, 0, 0] = 1
    B[0, 1, 1, 0] = 1
    return MPO([B] * n_qubits, strategy=strategy)


def x_mpo(
    n_qubits: int, a: float, dx: float, strategy: Strategy = DEFAULT_STRATEGY
) -> MPO:
    """x MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        x operator MPO.
    """
    MPO_x = []

    if n_qubits == 1:
        B = np.zeros((1, 2, 2, 1))
        B[0, 0, 0, 0] = a
        B[0, 1, 1, 0] = a + dx
        MPO_x.append(B)
        return MPO(MPO_x, strategy=strategy)
    else:
        for i in range(n_qubits):
            if i == 0:
                Bi = np.zeros((1, 2, 2, 2))
                Bi[0, 0, 0, 0] = 1
                Bi[0, 1, 1, 0] = 1
                Bi[0, 0, 0, 1] = a
                Bi[0, 1, 1, 1] = a + dx * 2 ** (n_qubits - 1)
                MPO_x.append(Bi)
            elif i == n_qubits - 1:
                Bf = np.zeros((2, 2, 2, 1))
                Bf[1, 0, 0, 0] = 1
                Bf[1, 1, 1, 0] = 1
                Bf[0, 1, 1, 0] = dx
                MPO_x.append(Bf)
            else:
                B = np.zeros((2, 2, 2, 2))
                B[0, 0, 0, 0] = 1
                B[0, 1, 1, 0] = 1
                B[1, 0, 0, 1] = 1
                B[1, 1, 1, 1] = 1
                B[0, 1, 1, 1] = dx * 2 ** (n_qubits - 1 - i)
                MPO_x.append(B)

        return MPO(MPO_x, strategy=strategy)


def x_to_n_mpo(
    n_qubits: int,
    a: float,
    dx: float,
    n: int,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """x^n MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    n: int
        Order of the x polynomial.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        x^n operator MPO.
    """
    # TODO: We have more efficient methods with polynomials now
    return MPOList([x_mpo(n_qubits, a, dx)] * n).join(strategy=strategy)


def p_mpo(n_qubits: int, dx: float, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """p MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        p operator MPO.
    """
    dk = 2 * np.pi / (dx * 2**n_qubits)
    MPO_p = []

    if n_qubits == 1:
        B = np.zeros((1, 2, 2, 1))
        B[0, 1, 1, 0] = dk * (1 - 2**n_qubits)
        MPO_p.append(B)
        return MPO(MPO_p, strategy=strategy)
    for i in range(n_qubits):
        if i == 0:
            Bi = np.zeros((1, 2, 2, 2))
            Bi[0, 0, 0, 0] = 1
            Bi[0, 1, 1, 0] = 1
            Bi[0, 1, 1, 1] = dk * 2 ** (n_qubits - 1) - dk * 2**n_qubits
            MPO_p.append(Bi)
        elif i == n_qubits - 1:
            Bf = np.zeros((2, 2, 2, 1))
            Bf[1, 0, 0, 0] = 1
            Bf[1, 1, 1, 0] = 1
            Bf[0, 1, 1, 0] = dk
            MPO_p.append(Bf)
        else:
            B = np.zeros((2, 2, 2, 2))
            B[0, 0, 0, 0] = 1
            B[0, 1, 1, 0] = 1
            B[1, 0, 0, 1] = 1
            B[1, 1, 1, 1] = 1
            B[0, 1, 1, 1] = dk * 2 ** (n_qubits - 1 - i)
            MPO_p.append(B)

    return MPO(MPO_p, strategy=strategy)


def p_to_n_mpo(
    n_qubits: int,
    dx: float,
    n: int,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """p^n MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    dx: float
        Spacing of the position interval.
    n: int
        Order of the x polynomial.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        p^n operator MPO.
    """
    # TODO: We have more efficient ways to do this now with polynomials
    return MPOList([p_mpo(n_qubits, dx)] * n).join(strategy=strategy)


def exponential_mpo(
    n: int,
    a: float,
    dx: float,
    c: complex = 1,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """exp(cx) MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    c: complex, default = 1
        Constant preceeding the x coordinate.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        exp(x) operator MPO.
    """
    MPO_exp = []
    if n == 1:
        B = np.zeros((1, 2, 2, 1), complex)
        B[0, 0, 0, 0] = np.exp(c * a)
        B[0, 1, 1, 0] = np.exp(c * (a + dx))
        MPO_exp.append(B)
        return MPO(MPO_exp, strategy=strategy)
    else:
        for i in range(n):
            if i == 0:
                Bi = np.zeros((1, 2, 2, 1), complex)
                Bi[0, 0, 0, 0] = np.exp(c * (a))
                Bi[0, 1, 1, 0] = np.exp(c * (a + dx * 2 ** (n - 1)))
                MPO_exp.append(Bi)
            elif i == n - 1:
                Bf = np.zeros((1, 2, 2, 1), complex)
                Bf[0, 0, 0, 0] = 1
                Bf[0, 1, 1, 0] = np.exp(c * dx)
                MPO_exp.append(Bf)
            else:
                B = np.zeros((1, 2, 2, 1), complex)
                B[0, 0, 0, 0] = 1
                B[0, 1, 1, 0] = np.exp(c * dx * 2 ** (n - 1 - i))
                MPO_exp.append(B)

        return MPO(MPO_exp, strategy=strategy)


def cos_mpo(n: int, a: float, dx: float, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """cos(x) MPO.S

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        cos(x) operator MPO.
    """
    exp1 = exponential_mpo(n, a, dx, c=+1j, strategy=strategy)
    exp2 = exponential_mpo(n, a, dx, c=-1j, strategy=strategy)
    cos_mpo = 0.5 * (exp1 + exp2)
    return cos_mpo.join(strategy=strategy)


def sin_mpo(n: int, a: float, dx: float, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
    """sin(x) MPO.

    Parameters
    ----------
    n_qubits: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    strategy: Strategy
        MPO strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPO
        sin(x) operator MPO.
    """
    exp1 = exponential_mpo(n, a, dx, c=+1j, strategy=strategy)
    exp2 = exponential_mpo(n, a, dx, c=-1j, strategy=strategy)
    sin_mpo = (-1j) * 0.5 * (exp1 - exp2)
    return sin_mpo.join(strategy=strategy)


def mpo_affine(
    mpo: MPO,
    orig: tuple[float, float],
    dest: tuple[float, float],
) -> MPO:
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    mpo_affine = a * mpo
    if abs(b) > np.finfo(np.float64).eps:
        I = MPO([np.ones((1, 2, 2, 1))] * len(mpo_affine))
        mpo_affine = MPOSum(mpos=[mpo_affine, I], weights=[1, b]).join()
    return mpo_affine


def mpo_cumsum(n: int) -> MPO:
    """Returns an MPO that computes the cumulative sum of an input MPS."""
    core_L = np.zeros((1, 2, 2, 2), dtype=np.float64)
    core_L[0, 0, 0, 0] = 1
    core_L[0, 1, 1, 0] = 1
    core_L[0, 1, 0, 1] = 1
    cores_bulk = []
    for _ in range(1, n - 1):
        core = np.zeros((2, 2, 2, 2), dtype=np.float64)
        core[0, 0, 0, 0] = 1
        core[0, 1, 1, 0] = 1
        core[0, 1, 0, 1] = 1
        core[1, :, :, 1] = 1
        cores_bulk.append(core)
    core_R = np.zeros((2, 2, 2, 1), dtype=np.float64)
    core_R[0, 0, 0, 0] = 1
    core_R[0, 1, 1, 0] = 1
    core_R[0, 1, 0, 0] = 1
    core_R[1, :, :, 0] = 1
    return MPO([core_L] + cores_bulk + [core_R])
