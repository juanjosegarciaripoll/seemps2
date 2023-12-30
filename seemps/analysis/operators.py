import numpy as np
from ..operators import MPO, MPOList
from ..state import Strategy, DEFAULT_STRATEGY
from typing import Union


def id_mpo(n_qubits: int, strategy=DEFAULT_STRATEGY):
    """Identity MPO.

    Arguments:
    ----------
    Parameters:
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
    return MPO([B for n_i in range(n_qubits)], strategy=strategy)


def x_mpo(n_qubits: int, a: float, dx: float, strategy=DEFAULT_STRATEGY):
    """x MPO.

    Parameters:
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
    strategy=DEFAULT_STRATEGY,
):
    """x^n MPO.

     Parameters:
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
    return MPOList([x_mpo(n_qubits, a, dx) for n_i in range(n)]).join(strategy=strategy)


def p_mpo(n_qubits: int, dx: float, strategy=DEFAULT_STRATEGY):
    """p MPO.

    Parameters:
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
    strategy=DEFAULT_STRATEGY,
):
    """p^n MPO.

    Parameters:
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
    return MPOList([p_mpo(n_qubits, dx) for n_i in range(n)]).join(
        strategy=strategy,
    )


def exponential_mpo(
    n: int,
    a: float,
    dx: float,
    c: Union[float, complex] = 1,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    """exp(cx) MPO.

    Parameters:
    ----------
    n_qubits: int
        Number of qubits.
    a: float
        Initial value of the position interval.
    dx: float
        Spacing of the position interval.
    c: float | complex, default = 1
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


def cos_mpo(n: int, a: float, dx: float, strategy=DEFAULT_STRATEGY):
    """cos(x) MPO.S

    Parameters:
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


def sin_mpo(n: int, a: float, dx: float, strategy=DEFAULT_STRATEGY):
    """sin(x) MPO.

    Parameters:
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
