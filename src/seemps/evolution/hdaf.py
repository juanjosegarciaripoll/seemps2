from __future__ import annotations
import numpy as np
from typing import Callable, Any
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..state.simplification import simplify
from ..analysis.mesh import Mesh, QuantizedInterval, mps_to_mesh_matrix
from ..analysis.cross.black_box import BlackBoxLoadMPS
from ..analysis.cross.cross_dmrg import cross_dmrg
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
    hdaf_order: int = 10,
) -> MPS | list[Any]:
    r"""Solve the Schrödinger equation via second-order Strang splitting.

    Evolves a state under the Hamiltonian

    .. math::
        H = -\frac{1}{2}\frac{d^2}{dx^2} + V(x)

    by alternating half-steps of the potential propagator
    :math:`e^{-i V(x) \delta t/2}` with a full kinetic step
    :math:`e^{i \delta t/2 \, d^2/dx^2}` approximated via HDAF.

    Parameters
    ----------
    potential_func : Callable[[np.ndarray], np.ndarray]
        Function representing the potential :math:`V(x)`.
    time : TimeSpan
        Integration interval. Only constant time steps (float or tuple) are supported.
    state : MPS
        Initial MPS wavefunction.
    a : float
        Left endpoint of the spatial interval.
    num_qubits : int
        Number of qubits for the discretization (:math:`2^N` grid points).
    dx : float
        Grid spacing.
    periodic : bool
        If True, the interval is half-open :math:`[a, b)` (periodic BCs).
        If False, it is closed :math:`[a, b]`.
    steps : int, default=1000
        Number of time steps.
    strategy : Strategy
        Truncation strategy for MPS simplification.
    callback : ODECallback | None
        Called after each step; if provided, returns a list of callback values.
    hdaf_order : int, default=10
        Order of the HDAF approximation for the kinetic propagator.

    Returns
    -------
    MPS | list[Any]
        Evolved state, or list of callback values if `callback` is provided.
    """
    size = 2**num_qubits
    b = a + dx * (size if periodic else size - 1)
    interval = QuantizedInterval(a, b, num_qubits, endpoint_right=not periodic)
    mesh = Mesh([interval])

    if isinstance(time, (float, int)):
        start, stop = 0.0, float(time)
    elif isinstance(time, tuple):
        start, stop = float(time[0]), float(time[1])
    else:
        raise ValueError(
            "split_step only supports constant time steps (float or tuple)."
        )
    dt = (stop - start) / steps

    # Potential half-step: exp(-i * V(x) * dt/2)
    def propagator_func(x: np.ndarray) -> np.ndarray:
        return np.exp(-1j * potential_func(x) * (dt / 2))

    physical_dimensions = [2] * num_qubits
    map_matrix = mps_to_mesh_matrix([num_qubits], base=2)
    black_box = BlackBoxLoadMPS(propagator_func, mesh, map_matrix, physical_dimensions)
    U_potential_mps = cross_dmrg(black_box).mps

    # Kinetic full-step: exp(-i * dt * K) with K = -1/2 d^2/dx^2
    U_kinetic_mpo = hdaf_mpo(
        num_qubits=num_qubits,
        dx=dx,
        M=hdaf_order,
        time=dt,
        derivative=0,
        periodic=periodic,
        strategy=strategy,
    )

    def evolve_for_dt(t: float, state: MPS, dt_step: float, strategy: Strategy) -> MPS:
        state = simplify(U_potential_mps * state, strategy)
        state = simplify(U_kinetic_mpo @ state, strategy)
        state = simplify(U_potential_mps * state, strategy)
        return state

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)
