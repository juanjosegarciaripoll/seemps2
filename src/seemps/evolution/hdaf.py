from __future__ import annotations
import numpy as np
from typing import Callable, Any
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..state.simplification import simplify
from ..analysis.mesh import Mesh, QuantizedInterval, mps_to_mesh_matrix
from ..analysis.cross.black_box import BlackBoxLoadMPS
from ..analysis.cross.cross_dmrg import cross_dmrg, CrossStrategyDMRG
from ..analysis.hdaf import hdaf_mpo
from .common import ode_solver, ODECallback, TimeSpan


def split_step(
    potential_func: Callable[[np.ndarray], np.ndarray],
    time: TimeSpan,
    state: MPS,
    a: float,
    num_qubits: int,
    dx: float,
    periodic: bool,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    hdaf_order: int = 30,
    itime: bool = False,
) -> MPS | list[Any]:
    """
    Implements a second-order Strang splitting time evolution for a Hamiltonian :math:`H = -\\frac{1}{2}\\frac{d^2}{dx^2} + V(x)`.

    Parameters
    ----------
    potential_func : Callable[[np.ndarray], np.ndarray]
        A function representing V(x).
    time : TimeSpan
        Integration interval. Only constant time steps (float or tuple) are supported.
    state : MPS
        The initial MPS wavefunction.
    a : float
        Start of the interval.
    num_qubits : int
        Number of qubits for the discretization.
    dx : float
        Step size.
    periodic : bool
        If True, the interval is open [a, b) (Periodic BCs).
        If False, it is closed [a, b] (Closed BCs).
    steps : int, default=1000
        Number of time steps.
    strategy : Strategy
        Truncation strategy for MPS simplification.
    callback : ODECallback | None
        Function called after each step.
    hdaf_order : int, default=30
        Order of the HDAF approximation for the kinetic propagator.
    itime : bool, default=False
        Whether to perform imaginary time evolution.

    Returns
    -------
    MPS | list[Any]
        The evolved state or callback results.
    """
    # Build space
    size = 2**num_qubits
    if periodic:
        b = a + dx * size
    else:
        b = a + dx * (size - 1)

    interval = QuantizedInterval(a, b, num_qubits, endpoint_right=not periodic)
    mesh = Mesh([interval])

    if itime:
        factor = 1.0
    else:
        factor = 1j

    # Determine dt and steps
    if isinstance(time, (float, int)):
        start, stop = 0.0, float(time)
    elif isinstance(time, tuple):
        start, stop = float(time[0]), float(time[1])
    else:
        raise ValueError(
            "split_step only supports constant time steps (float or tuple)."
        )

    dt = (stop - start) / steps

    # Potential Propagator exp(-factor * V(x) * dt / 2)
    def propagator_func(x):
        return np.exp(-factor * potential_func(x) * dt / 2)

    physical_dimensions = [2] * num_qubits
    map_matrix = mps_to_mesh_matrix([num_qubits], base=2)
    black_box = BlackBoxLoadMPS(propagator_func, mesh, map_matrix, physical_dimensions)
    cross_results = cross_dmrg(black_box, CrossStrategyDMRG())
    U_potential_mps = cross_results.mps

    # Kinetic Propagator exp(-factor * T * dt) with T = -0.5 * d^2/dx^2
    if isinstance(factor, complex) and factor.imag != 0:
        hdaf_time = dt  # Imaginary time
    else:
        hdaf_time = -1j * dt  # Real time

    U_kinetic_mpo = hdaf_mpo(
        num_qubits=num_qubits,
        dx=dx,
        M=hdaf_order,
        time=hdaf_time,
        derivative=0,
        periodic=periodic,
        strategy=strategy,
    )

    def evolve_for_dt(
        t: float,
        state: MPS,
        factor: complex | float,
        dt_step: float,
        strategy: Strategy,
    ) -> MPS:
        # Apply half-step potential
        state = U_potential_mps * state
        state = simplify(state, strategy)

        # Apply full-step kinetic
        state = U_kinetic_mpo @ state
        state = simplify(state, strategy)

        # Apply half-step potential
        state = U_potential_mps * state
        state = simplify(state, strategy)

        return state

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)
