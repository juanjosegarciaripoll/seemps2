from __future__ import annotations
import numpy as np
from ..analysis.operators import id_mpo
from ..solve import cgs_solve
from ..operators import MPO, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, Strategy, simplify
from .common import ode_solver, ODECallback, TimeSpan


def euler(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
):
    r"""Solve ``d|state>/dt = L|state>`` using the Euler method.

    Integrates a linear ODE using the update

    .. math::
        \psi(t_{n+1}) = \psi(t_{n}) + \delta t L \psi(t_{n}).

    The `time` denotes the integration interval.

    - If it is a single number `T`, the initial condition is :math:`t=0`
      and the evolution proceeds in steps of :math:`\delta{t}=T/N`
      where `N=steps`.

    - If `time` is a tuple, it contains the initial and final time,
      and the number of integration steps is deduced from `N=steps`
      as :math:`\delta{t}=T/N`

    - If `time` is a sequence of numbers, starting with the initial
      condition, and progressing in time steps `time[n+1]-time[n]`.

    The Euler algorithm is a very bad integrator and is offered only for
    illustrative purposes.

    Parameters
    ----------
    L : MPO
        Linear operator in MPO form.
    time : Real | tuple[Real, Real] | Sequence[Real]
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Callable[[float, MPS], Any] | None
        A callable called after each iteration (defaults to None).

    Returns
    -------
    result : MPS | list[Any]
        Final state after evolution or values collected by callback
    """

    def evolve_for_dt(
        t: float,
        state: MPS,
        dt: float,
        strategy: Strategy,
    ) -> MPS:
        return simplify(state + dt * (L @ state), strategy=strategy)

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)


def euler2(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
):
    r"""Solve ``d|state>/dt = L|state>`` using the 2nd order Euler method.

    Implements a two-step integration method,

    .. math::
        \xi(t_{n}) = 2 \psi(t_{n}) + \delta t L \psi(t_{n})
        \psi(t_{n+1}) = \psi(t_{n}) + (\delta t/2) L \xi(t_{n})

    The Euler algorithm is a very bad integrator and is offered only for
    illustrative purposes. See :func:`euler` to understand the parameters
    and the function's output.
    """

    def evolve_for_dt(
        t: float,
        state: MPS,
        dt: float,
        strategy: Strategy,
    ) -> MPS:
        xi = simplify(2.0 * state + dt * (L @ state), strategy=strategy)
        return simplify(state + (0.5 * dt) * (L @ xi), strategy=strategy)

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)


def implicit_euler(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    tolerance: float = 1e-10,
    maxiter_cgs: int = 50,
):
    r"""Solve ``d|state>/dt = L|state>`` using a 2nd order implicit Euler method.

    See :func:`seemps.evolution.euler` for a description of the missing
    function arguments and the function's output

    Parameters
    ----------
    tol_cgs: float
        Tolerance of the CGS algorithm.
    maxiter_cgs: int
        Maximum number of iterations of the CGS algorithm.
    """
    last_dt: float = np.inf
    A: MPO | None = None
    B: MPO | None = None
    id = id_mpo(state.size, strategy=strategy)

    def evolve_for_dt(
        t: float,
        state: MPS,
        dt: float,
        strategy: Strategy,
    ) -> MPS:
        nonlocal A, B, last_dt
        if last_dt != dt or A is None or B is None:
            last_dt = dt
            A = MPOSum(mpos=[id, L], weights=[1, -0.5 * dt]).join(strategy=strategy)
            B = MPOSum(mpos=[id, L], weights=[1, 0.5 * dt]).join(strategy=strategy)
            # TODO: Fixed tolerance criteria
            state, _ = cgs_solve(
                A,
                B @ state,
                strategy=strategy,
                tolerance=tolerance,
                maxiter=maxiter_cgs,
            )
        return state

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)
