from __future__ import annotations
import numpy as np
from typing import Union, Optional, Callable
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..operators import MPO
from ..truncate import simplify
from ..typing import Vector


def euler(
    H: MPO,
    t_span: Union[float, tuple[float, float], Vector],
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Optional[Callable] = None,
    itime: bool = False,
):
    r"""Solve a Schrodinger equation using the Euler method.

    Integrates the Schrodinger equation in real or imaginary time using
    a first order Euler method. The equation is defined as
    .. math::
        i\frac{d}{dt}\psi = H(t) \psi
    in real time and as
    .. math::
        \frac{d}{dt}\psi = -H(t) \psi
    in imaginary time evolution.

    The integration algorithm is very simple. It is
    .. math::
        \psi(t_{n+1}) = \psi(t_{n}) - i \delta t H \psi(t_{n})
    for real time and
    .. math::
        \psi(t_{n+1}) = \psi(t_{n}) - \delta t H \psi(t_{n})
    for imaginary time evolution. The integration step is deduced from
    the arguments as explained below.

    The `t_span` denotes the integration interval.
    * If it is a single number `T`, the initial condition is ..math::`t=0`
      the evolution proceeds in steps of :math:`\delta{t}=T/N`
      where `N=steps`.
    * If `t_span` is a tuple, it contains the initial and final time,
      and the number of integration steps is deduced from `N=steps`
      as :math:`\delta{t}=T/N`
    * If `t_span` is a sequence of numbers, starting with the initial
      condition, and progressing in time steps `t_span[n+1]-t_span[n]`.

    The Euler algorithm is a very bad integrator and is offered only for
    illustrative purposes.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    t_span : float | tuple[float, float] | Vector
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Optional[Callable[[float, MPS], Any]]
        A callable called after each iteration (defaults to None).
    itime : bool, default = False
        Whether to solve the imaginary time evolution problem.

    Results
    -------
    result : MPS | list[Any]
        Final state after evolution or values collected by callback
    """
    if isinstance(t_span, (int, float)):
        t_span = (0.0, t_span)
    if len(t_span) == 2:
        t_span = np.linspace(t_span[0], t_span[1], steps + 1)
    factor: float | complex
    if itime:
        factor = 1
        strategy = strategy.replace(normalize=True)
    else:
        factor = 1j
    last_t = t_span[0]
    output = []
    for t in t_span:
        if t != last_t:
            idt = factor * (t - last_t)
            state = simplify(state - idt * (H @ state), strategy=strategy)
        if callback is not None:
            output.append(callback(t, state))
        last_t = t
    if callback is None:
        return state
    else:
        return output


def euler2(
    H: MPO,
    t_span: Union[float, tuple[float, float], Vector],
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Optional[Callable] = None,
    itime: bool = False,
):
    r"""Solve a Schrodinger equation using the 2nd order Euler method.

    Implements a two-step integration method. In imaginary time this is
    .. math::
        \xi(t_{n}) = 2 \psi(t_{n}) - i \delta t H \psi(t_{n})
        \psi(t_{n+1}) = \psi(t_{n}) - (i \delta t/2) H \xi(t_{n})

    The Euler algorithm is a very bad integrator and is offered only for
    illustrative purposes. See :function:`euler` to understand the parameters.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    t_span : float | tuple[float, float] | Vector
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Optional[Callable[[float, MPS], Any]]
        A callable called after each iteration (defaults to None).
    itime : bool, default = False
        Whether to solve the imaginary time evolution problem.

    Results
    -------
    result : MPS | list[Any]
        Final state after evolution or values collected by callback
    """
    if isinstance(t_span, (int, float)):
        t_span = (0.0, t_span)
    if len(t_span) == 2:
        t_span = np.linspace(t_span[0], t_span[1], steps + 1)
    factor: float | complex
    if itime:
        factor = 1
        strategy = strategy.replace(normalize=True)
    else:
        factor = 1j
    last_t = t_span[0]
    output = []
    for t in t_span:
        if t != last_t:
            idt = factor * (t - last_t)
            xi = simplify(2.0 * state - idt * (H @ state), strategy=strategy)
            state = simplify(state - (0.5 * idt) * (H @ xi), strategy=strategy)
        if callback is not None:
            output.append(callback(t, state))
        last_t = t
    if callback is None:
        return state
    else:
        return output
