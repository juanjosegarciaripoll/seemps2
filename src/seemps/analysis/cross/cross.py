import numpy as np
from copy import deepcopy
from typing import Callable
from time import perf_counter
from dataclasses import dataclass

from ...state import MPS, random_mps
from ...typing import *
from ...tools import DEBUG, log
from ..mesh import Mesh, mps_to_mesh_matrix
from ..sampling import random_mps_indices, evaluate_mps
from .maxvol import maxvol


@dataclass
class Cross:
    state: MPS
    I_g: np.ndarray
    func: Callable
    mesh: Mesh
    T: np.ndarray

    """
    A data structure containing a Cross object, represented by a MPS and a collection
    of initial multi-indices; a multivariate function and a mesh where it is discretized;
    and a linear transformation T between the multiindices of the MPS and the multiindices
    of the mesh.
    """

    def __post_init__(self):
        self.sites = len(self.state)
        self.I_s, self.I_le, _ = empty_multiindices(self.state)
        self.maxbond = max(self.state.bond_dimensions())
        self.evals = 0
        self.sweep = 0

    # TODO: Consider caching the indices to speed up the sampling
    def sample_func(self, mps_indices: np.ndarray) -> np.ndarray:
        """
        Samples the function on the given MPS indices.
        """
        return self.func(self.mesh[mps_indices @ self.T])

    def sample_state(self, mps_indices: np.ndarray) -> np.ndarray:
        """
        Samples the MPS on the given MPS indices.
        """
        return evaluate_mps(self.state, mps_indices)

    def sample_fiber(self, k: int) -> np.ndarray:
        """
        Samples the k-th tensor fiber of the underlying MPS from the multiindices
        centered at site k.
        Namely, samples $A(J_{\le k-1}, i_k, J_{\gk})$ (see dolgov2020).
        """
        i_le, i_s, i_g = self.I_le[k], self.I_s[k], self.I_g[k + 1]
        r_le = i_le.shape[0] if i_le.size > 0 else 1
        r_g = i_g.shape[0] if i_g.size > 0 else 1
        s = i_s.shape[0]
        mps_indices = np.kron(
            np.kron(np.ones((r_g, 1), dtype=int), i_s), np.ones((r_le, 1), dtype=int)
        )
        if i_le.size > 0:
            mps_indices = np.hstack(
                (np.kron(np.ones((s * r_g, 1), dtype=int), i_le), mps_indices)
            )
        if i_g.size > 0:
            mps_indices = np.hstack(
                (mps_indices, np.kron(i_g, np.ones((r_le * s, 1), dtype=int)))
            )
        return self.sample_func(mps_indices).reshape((r_le, s, r_g), order="F")  # type: ignore


@dataclass
class CrossStrategy:
    maxiter: int = 50
    maxbond: int = 100
    tol_error: float = 1e-10
    tol_variation: float = 1e-14
    min_bond_change: int = 1
    max_bond_change: int = 100
    rng: Optional[np.random.Generator] = np.random.default_rng()
    order: str = "A"
    physical_dimension: int = 2
    starting_bond: int = 1
    """
    Dataclass containing methods and hyperparameters of the tensor 
    cross-interpolation algorithm. The methods can be defined at custom
    by subclassing this class.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of sweeps allowed.
    maxbond : int, default=100
        Maximum bond dimension allowed for the MPS.
    tol_error : float, default=1e-10
        Tolerance with respect to the results of the 'error' method.
    tol_variation : float, default=1e-14
        Tolerance with respect to the results of the 'variation' method.
    min_bond_change : int, default=1
        Minimum bond change allowed for the maxvol algorithm per half-sweep.
    max_bond_change : int, default=np.inf
        Maximum bond change allowed for the maxvol algorithm per half-sweep.
    rng : np.random.Generator, default=None
        The random number generator to used to initialize the initial multiindices.
        If None, uses Numpy's default generator without any predefined seed.
    order : str
        The order of the MPS determining how the tensor fibers are sampled.
    physical_dimension : int
        The physical dimension of the MPS, determining how the fibers are sampled.
    starting_bond : int
        The starting bond dimensions of the initial MPS.
    """

    def error(self, cross: Cross) -> float:
        """
        Computes the error of the interpolated state.
        By default, comptues the maximum error between randomly sampled
        subtensors of the function and the state.
        """
        if not hasattr(self, "random_indices"):
            self.random_indices = random_mps_indices(cross.state, rng=self.rng)
            self.func_samples = cross.sample_func(self.random_indices).reshape(-1)
        state_samples = cross.sample_state(self.random_indices)
        return np.max(np.abs(state_samples - self.func_samples))

    def variation(self, cross: Cross) -> float:
        """
        Computes a value between the interpolated state at one iteration and the
        updated state at a different iteration.
        By default, compares the norm-2 of the difference between the two states.
        """
        if not hasattr(self, "previous_data"):
            self.previous_data = deepcopy(cross.state._data)  # Expensive
            return np.Inf
        previous_state = MPS(self.previous_data)
        state_variation = (cross.state - previous_state).join()
        variation = abs(state_variation.norm()) / previous_state.norm()
        self.previous_data = deepcopy(cross.state._data)
        return variation

    def bond_update_strategy(self, cross: Cross, ltr: bool) -> tuple:
        """
        Determines the strategy for the runtime updates of the bond dimension.
        By default, stops updating the bond dimension when it reaches the
        threshold defined by the CrossStrategy maxbond parameter.
        """
        if cross.maxbond < self.maxbond:
            min_r = self.min_bond_change
            max_r = min(self.maxbond - cross.maxbond, self.max_bond_change)
        else:
            min_r = 0
            max_r = 0
        return min_r, max_r

    def converged(self, cross: Cross, error: float, variation: float) -> bool:
        """
        Determines the convergence criteria followed by the algorithm.
        Here, both the error, variation and maxbond are considered.
        """
        converged = False
        message = f"Maximum number of sweeps {self.maxiter} reached"
        if not hasattr(self, "maxbond_prev"):
            self.maxbond_prev = 0
        if error < self.tol_error:
            converged = True
            message = f"State converged within tolerance {self.tol_error}"
        elif variation < self.tol_variation:
            converged = True
            message = f"Stationary state reached within tolerance {self.tol_variation}"
        elif cross.maxbond >= self.maxbond:
            converged = True
            message = f"Maxbond reached above the threshold {self.maxbond}"
        elif cross.maxbond <= self.maxbond_prev:
            converged = True
            message = (
                f"Maxbond reached for the given state with {len(cross.state)} qubits"
            )
        self.maxbond_prev = cross.maxbond
        return converged, message


@dataclass
class CrossResults:
    state: MPS
    error: float
    converged: bool
    message: str
    trajectories: Optional[VectorLike] = None
    """Results from tensor cross-interpolation.

    Parameters
    ----------
    state : MPS
        The estimate for the ground state.
    error : float
        Estimate for the error committed by the interpolation.
    converged : bool
        True if the algorithm has converged according to the given tolerances.
    message : str
        Message explaining why the algorithm stoped, both when it converged,
        and when it did not.
    trajectories : Optional[Vector]
        Vector of additional information for each sweep of the algorithm.
    """


def cross_interpolation(
    func: Callable,
    mesh: Mesh,
    cross_strategy: Optional[CrossStrategy] = None,
    initial_state: Optional[MPS] = None,
    callback: Optional[Callable] = None,
) -> CrossResults:
    """
    Performs the tensor cross-interpolation (TCI) algorithm on a vectorized
    function discretized on a mesh to obtain its MPS representation.
    Uses the Alternating Least Squares (ALS) scheme by means of the rectangular
    maxvol algorithm. Usually, the bond dimension of the resulting MPS is greatly
    overestimated, so it requires an additional simplification step.

    Parameters
    ----------
    func : Callable[[numpy.ndarray], numpy.ndarray]
        The vectorized function to be approximated as a MPS. It takes as input
        an N+1 dimensional array, `X[...,d]`, where the index 'd' runs over the
        `d` dimensions of the mesh, and returns an N-dimensional array with the
        function evaluated on those coordinates.
    mesh : Mesh
        The mesh of points where the function is discretized.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        The strategy configuration for the algorithm.
    initial_state : MPS, default=None
        An initial approximation for the algorithm, to sample a collection of
        initial multiindices providing a better starting point. If None, the
        algorithm starts at a random starting point.
    callback : Optional[Callable], default=None
        An optional callback function called on the state after each sweep.

    Returns
    -------
    results: CrossResults
        The results of the cross interpolation process, including the
        final MPS state, error, convergence status and message. Optionally,
        returns other information about the trajectories of a collection of
        magnitudes.
    """
    # Avoid default argument initialization pitfall
    # (the cross_strategy does not reinitialize for different calls of cross_interpolation()
    #  and, for example makes test fail)
    if cross_strategy == None:
        cross_strategy = CrossStrategy()
    # Initialize Cross data structure
    base = cross_strategy.physical_dimension
    sites_per_dimension = [int(np.emath.logn(base, s)) for s in mesh.dimensions]
    if initial_state is None:
        sites = sum(sites_per_dimension)
        state = random_mps([base] * sites, cross_strategy.starting_bond)
        I_g = random_multiindices(state, cross_strategy.rng)
    else:
        state = initial_state
        I_g = sampled_multiindices(state)
    if not all(base**n == N for n, N in zip(sites_per_dimension, mesh.dimensions)):
        raise ValueError(f"The mesh size cannot be quantized with dimension {base}")
    T = mps_to_mesh_matrix(sites_per_dimension, cross_strategy.order)
    cross = Cross(state, I_g, func, mesh, T)

    # Optimize Cross until convergence
    if DEBUG:
        log(f"Initial TT-Cross state: maxbond = {cross.maxbond:3d}")
    for i in range(cross_strategy.maxiter):
        start_time = perf_counter()
        sweep(cross, cross_strategy, ltr=True)
        sweep(cross, cross_strategy, ltr=False)
        end_time = perf_counter()
        time = end_time - start_time

        error = cross_strategy.error(cross)
        variation = cross_strategy.variation(cross)

        if DEBUG:
            log(
                f"Results after TT-Cross sweep {1+i:3d}: error={error:.15e}, "
                f"variation={variation:.15e}, "
                f"maxbond={cross.maxbond:3d}, "
                f"time = {time:5f}, "
                f"evals = {cross.evals:8d}"
            )

        converged, message = cross_strategy.converged(cross, error, variation)
        if converged:
            break

        if callback is not None:
            callback(cross)

    log(message)
    return CrossResults(
        state=cross.state,
        error=error,
        converged=converged,
        message=message,
    )


def sweep(cross: Cross, cross_strategy: CrossStrategy, ltr: bool) -> None:
    """
    Performs a sweep of the tensor cross-interpolation algorithm on the
    Cross data structure, optimizing it inplace. Optimizes each tensor of
    the MPS following the maxvol principle on an Alternating Least Squares
    (ALS) scheme.
    """
    min_r, max_r = cross_strategy.bond_update_strategy(cross, ltr)
    if ltr:
        for k in range(cross.sites):
            fiber = cross.sample_fiber(k)
            cross.evals += fiber.size
            r_le, s, r_g = fiber.shape
            Q, R = np.linalg.qr(fiber.reshape(r_le * s, r_g, order="F"))
            I_maxvol, B = maxvol(Q, min_r, max_r)
            cross.state[k] = B.reshape(r_le, s, -1, order="F")
            stacked_indices = np.hstack(
                (
                    np.kron(np.ones((s, 1), dtype=int), cross.I_le[k]),
                    np.kron(cross.I_s[k], np.ones((r_le, 1), dtype=int)),
                )
            )
            cross.I_le[k + 1] = stacked_indices[I_maxvol, :]
        cross.state[k] = np.tensordot(cross.state[k], Q[I_maxvol, :] @ R, 1)
    else:
        for k in reversed(range(cross.sites)):
            fiber = cross.sample_fiber(k)
            cross.evals += fiber.size
            r_le, s, r_g = fiber.shape
            Q, R = np.linalg.qr(fiber.reshape(r_le, s * r_g, order="F").T)
            I_maxvol, B = maxvol(Q, min_r, max_r)
            cross.state[k] = B.T.reshape(-1, s, r_g, order="F")
            stacked_indices = np.hstack(
                (
                    np.kron(np.ones((r_g, 1), dtype=int), cross.I_s[k]),
                    np.kron(cross.I_g[k + 1], np.ones((s, 1), dtype=int)),
                )
            )
            cross.I_g[k] = stacked_indices[I_maxvol, :]
        cross.state[0] = np.tensordot((Q[I_maxvol, :] @ R).T, cross.state[0], 1)
    cross.sweep += 1
    cross.maxbond = max(cross.state.bond_dimensions())


# TODO: We have to make I_s an ArrayLike because it is used as a
# list of as arrays later on. We should fix the TT Cross algorithm
# to avoid using lists at some point
def random_multiindices(
    state: MPS, rng: np.random.Generator = np.random.default_rng()
) -> list[np.ndarray]:
    """
    Returns a list of random multiindices used to initialize of a Cross object.
    Each index is chosen randomly from the physical dimensions of each tensor of the
    MPS and then progressively nested from the MPS right end to the left.
    """
    I_s, _, I_g = empty_multiindices(state)
    maxbond = max(state.bond_dimensions())
    for i, i_s in reversed(list(enumerate(I_s))):
        choice = rng.choice(i_s, size=maxbond, replace=True).reshape(-1, 1)
        if i == len(I_s) - 1:
            I_g.insert(0, choice)
        else:
            I_g.insert(0, np.hstack([choice, I_g[0]]))
    return I_g


def sampled_multiindices(state: MPS) -> list[np.ndarray]:
    """
    Returns a list of sampled multiindices used to initialize of a Cross object.
    The multiindices are computed by a cross-interpolation sweep.
    """
    I_s, I_le, I_g = empty_multiindices(state)
    R = np.ones((1, 1))
    for k in range(len(state)):
        fiber = np.tensordot(R, state[k], 1)
        r_le, s, r_g = fiber.shape
        Q, R = np.linalg.qr(fiber.reshape(r_le * s, r_g))
        I_maxvol, _ = maxvol(Q, 0, 0)
        R = Q[I_maxvol, :] @ R
        stacked_indices = np.hstack(
            (
                np.kron(np.ones((s, 1), dtype=int), I_le[k]),
                np.kron(I_s[k], np.ones((r_le, 1), dtype=int)),
            )
        )
        I_le[k + 1] = stacked_indices[I_maxvol, :]
    R = np.ones((1, 1))
    for k in reversed(range(len(state))):
        fiber = np.tensordot(state[k], R, 1)
        r_le, s, r_g = fiber.shape
        Q, R = np.linalg.qr(fiber.reshape(r_le, s * r_g).T)
        I_maxvol, _ = maxvol(Q, 0, 0)
        R = (Q[I_maxvol, :] @ R).T
        stacked_indices = np.hstack(
            (
                np.kron(np.ones((r_g, 1), dtype=int), I_s[k]),
                np.kron(I_g[k + 1], np.ones((s, 1), dtype=int)),
            )
        )
        I_g[k] = stacked_indices[I_maxvol, :]
    return I_g


def empty_multiindices(state: MPS):
    """
    Returns the multiindices corresponding to the physical dimensions of the state,
    as well as the empty sets of left and right multiindices.
    """
    sites = len(state)
    I_s = [np.arange(s).reshape(-1, 1) for s in state.physical_dimensions()]
    I_le = [np.array([], dtype=int)] * (sites + 1)
    I_g = [np.array([], dtype=int)] * (sites + 1)
    return I_s, I_le, I_g
