from __future__ import annotations
from .typing import Operator, Weight, Vector, to_dense_operator
import numpy as np
from .state.environments import (
    _begin_environment,
    _end_environment,
    _update_left_environment,
)
from .state.mps import MPS
from .operators import MPO


def expectation1(state: MPS, O: Operator, i: int) -> Weight:
    """Compute the expectation value :math:`\\langle\\psi|O_i|\\psi\\rangle`
    of an operator O acting on the `i`-th site

    Parameters
    ----------
    state : MPS
        Quantum state :math:`\\psi` used to compute the expectation value.
    O : Operator
        Local observable acting onto the mps.`i`-th subsystem
    i : int
        Index of site, in the range `[0, state.size)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    return state.expectation1(O, i)


def expectation2(
    state: MPS, O: Operator, Q: Operator, i: int, j: int | None = None
) -> Weight:
    """Compute the expectation value :math:`\\langle\\psi|O_i Q_j|\\psi\\rangle`
    of two operators `O` and `Q` acting on the `i`-th and `j`-th subsystems.

    Parameters
    ----------
    state : MPS
        Quantum state :math:`\\psi` used to compute the expectation value.
    O, Q : Operator
        Local observables
    i : int
    j : int, default=`i+1`
        Indices of sites, in the range `[0, state.size)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    return state.expectation2(O, Q, i, j)


def all_expectation1(state: MPS, O: list[Operator] | Operator) -> Vector:
    """Vector of expectation values :math:`v_i = \\langle\\psi|O_i|\\psi\\rangle`
    of local operators acting on individual sites of the MPS.

    Parameters
    ----------
    state: MPS
        State :math:`\\psi` onto which the expectation values are computed.
    operator : Operator | list[Operator]
        If `operator` is an observable, it is applied on each possible site.
        If it is a list, the expectation value of `operator[i]` is computed
        on the i-th site.

    Returns
    -------
    Vector
        Numpy array of expectation values.
    """
    return state.all_expectation1(O)


def product_expectation(state: MPS, operator_list: list[Operator]) -> Weight:
    """Expectation value of a product of local operators
    :math:`\\langle\\psi|O_0 O_1 \\cdots O_{N-1}|\\psi\\rangle`.

    Parameters
    ----------
    state : MPS
        State :math:`\\psi` onto which the expectation values are computed.
    operator_list : list[Operator]
        List of operators, with the same length `len(operator_list) == len(state)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    assert len(state) == len(operator_list)
    # TODO: Choose contraction order based on whether the state is
    # in a given canonical order or another
    rho = _begin_environment()
    for Ai, opi in zip(state, operator_list):
        rho = _update_left_environment(
            Ai.conj(), np.matmul(to_dense_operator(opi), Ai), rho
        )
    return _end_environment(rho)


def mpo_expectation(state: MPS, operator: MPO) -> Weight:
    return operator.expectation(state)
