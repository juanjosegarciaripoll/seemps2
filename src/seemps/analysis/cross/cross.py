from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from time import perf_counter
from math import prod
import numpy as np
import dataclasses
from typing import TypeAlias

from ...state import MPS
from ...tools import SEED, Logger, make_logger
from ...typing import Vector, Matrix, Tensor3, Tensor4, Natural
from ..evaluation import random_mps_indices, evaluate_mps
from .black_box import BlackBox, BlackBoxLoadMPS
from .kernels import combine_indices


@dataclasses.dataclass
class CrossStrategy:
    tol: float = 1e-8
    num_samples: Natural = 2**10
    error_norm: float = np.inf
    error_relative: bool = False
    range_iters: tuple[Natural, Natural] = (1, 200)
    range_max_bonds: tuple[Natural, Natural] = (1, 1000)
    max_time: float | None = None
    max_evals: Natural | None = None
    cache_evals: bool = dataclasses.field(default=False, kw_only=True)
    rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng(SEED)
    )
    """
    Abstract dataclass containing the base parameters for tensor cross interpolation.

    See specializations 
    :class:`~seemps.analysis.cross.CrossStrategyDMRG`,
    :class:`~seemps.analysis.cross.CrossStrategyGreedy`,
    :class:`~seemps.analysis.cross.CrossStrategyMaxvol`,
    :class:`~seemps.analysis.cross.CrossStrategyPRRLU`.

    Parameters
    ----------
    tol : float, default=1e-12
        Tolerance for the sampled error.
    num_samples : Natural, default=1024
        Number of function samples to evaluate the error.
    error_norm : float, default=np.inf
        L^p norm used for computing the error.
    error_relative : bool, default=False
        Whether to compute the absolute or relative error.
    range_iters : tuple[Natural, Natural], default=(1, 200)
        Range of iterations (half-sweeps) allowed.
    range_max_bonds : tuple[Natural, Natural], default=(1, 1000)
        Range of MPS maximum bond dimension allowed.
    max_time : float | None, default=None
        Maximum computation time allowed.
    max_evals : Natural | None, default=None
        Maximum number of evaluations allowed (cache misses).
    cache_evals : bool, default=False
        Cache black-box evaluations. Useful for expensive deterministic functions.
    rng : np.random.Generator, default=`numpy.random.default_rng(seemps.tools.SEED)`
        Random number generator used to initialize the algorithm and sample the error.
    """

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.range_iters[0] <= 0:
            raise ValueError("range_iters lower bound must be positive")
        if self.range_max_bonds[0] <= 0:
            raise ValueError("range_max_bonds lower bound must be positive")

    @abstractmethod
    def make_interpolator(
        self, black_box: BlackBox, initial_points: Matrix | None = None
    ) -> CrossInterpolation:
        pass


IndexMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer]]
IndexVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer]]
IndexSlice: TypeAlias = np.intp | IndexVector | slice


class CrossInterpolation:
    """Auxiliar base class for TCI used to keep track of required information."""

    black_box: BlackBox
    sites: int
    I_l: list[np.ndarray]
    I_g: list[np.ndarray]
    I_s: list[np.ndarray]
    mps: MPS
    two_sweeps_required: bool
    iteration_range: range
    evals: int
    cache_evals: bool
    cache: dict[int | tuple[int, ...], np.generic]

    def __init__(
        self,
        black_box: BlackBox,
        initial_points: Matrix | None = None,
        two_sweeps_required: bool = False,
        two_site_algorithm: bool = True,
        cache_evals: bool = False,
    ):
        self.black_box = black_box
        self.cache_evals = cache_evals
        self.cache = {}
        self.evals = 0
        self.sites = len(black_box.physical_dimensions)
        if initial_points is None:
            initial_points = np.zeros(self.sites, dtype=int)
        self.I_l, self.I_g = self.points_to_indices(initial_points)
        self.I_s = [np.arange(s).reshape(-1, 1) for s in black_box.physical_dimensions]
        self.mps = MPS([np.ones((1, s, 1)) for s in black_box.physical_dimensions])
        self.two_sweeps_required = two_sweeps_required
        self.iteration_range = range(
            self.sites - 1 if two_site_algorithm else self.sites
        )

    @abstractmethod
    def update(self, k: int, left_to_right: bool) -> None:
        pass

    def sample(self, *indices: IndexMatrix) -> Vector:
        """Evaluate the black box on the Cartesian product of index blocks.

        This method is used instead of indexing the black box directly.
        It allows counting evaluations, caching them when `cache_evals` is set,
        and implementing optimizations for specific black boxes.
        """
        if not self.cache_evals:
            self.evals += prod(len(block) for block in indices)
            if isinstance(self.black_box, BlackBoxLoadMPS):
                values = self.black_box.evaluate_blocks(*indices)  # Much faster
            else:
                values = self.black_box[combine_indices(*indices)]
            return np.asarray(values).reshape(-1)

        # Evaluate only the points not in the cache, each one once
        points = combine_indices(*indices)
        keys = list(map(tuple, points.tolist()))
        missing = {key: n for n, key in enumerate(keys) if key not in self.cache}
        if missing:
            values = self.black_box[points[list(missing.values())]]
            self.cache.update(zip(missing, np.asarray(values).reshape(-1)))
            self.evals += len(missing)
        return np.array([self.cache[key] for key in keys])

    def sample_fiber(self, k: int) -> Tensor3:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        return self.sample(i_l, i_s, i_g).reshape((len(i_l), len(i_s), len(i_g)))

    def sample_superblock(self, k: int) -> Tensor4:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        return self.sample(i_l, i_s1, i_s2, i_g).reshape(
            (len(i_l), len(i_s1), len(i_s2), len(i_g))
        )

    @staticmethod
    def points_to_indices(points: Matrix) -> tuple[list[Matrix], list[Matrix]]:
        if points.ndim == 1:
            points = points.reshape(1, -1)
        sites = points.shape[1]
        I_l = [points[:, :k] for k in range(sites)]
        I_g = [points[:, (k + 1) : sites] for k in range(sites)]
        return I_l, I_g


@dataclasses.dataclass
class CrossIterationResults:
    error: float
    bonds: Sequence[int]
    time: float
    evaluations: int

    @property
    def max_bond_dimension(self) -> int:
        return max(self.bonds)


class CrossResults:
    """
    Dataclass containing the results from TCI. Keeps track of values for every iteration (half-sweep).

    Parameters
    ----------
    mps : MPS
        The resulting MPS interpolation of the black-box function.
    errors : Vector
        Vector of error values.
    bonds : Matrix
        Matrix of intermediate bond dimensions.
    times : Vector
        Vector of cumulative computation times.
    evals : Vector
        Vector of cumulative function evaluations.
    """

    mps: MPS
    _points: list[CrossIterationResults]
    _start: float

    def __init__(self, mps: MPS):
        self.mps = mps
        self._points = []
        self._start = perf_counter()

    @property
    def errors(self) -> np.ndarray:
        return np.asarray([r.error for r in self._points])

    @property
    def bonds(self) -> np.ndarray:
        return np.asarray([r.bonds for r in self._points])

    @property
    def times(self) -> np.ndarray:
        return np.asarray([r.time for r in self._points])

    @property
    def evals(self) -> np.ndarray:
        return np.asarray([r.evaluations for r in self._points])

    def update(
        self,
        mps: MPS,
        error: float,
        bonds: Sequence[int],
        evaluations: int,
    ):
        self.mps = mps
        self._points.append(
            CrossIterationResults(
                error, bonds, perf_counter() - self._start, evaluations
            )
        )

    def get_result(self, index: int) -> CrossIterationResults:
        return self._points[index]


class CrossError:
    """
    Auxiliar base class for TCI used to compute the sampled error between the function and the
    MPS at every iteration using the sampled Lp norm and caching intermediate results for efficiency.
    """

    error_norm: float
    num_samples: int
    error_relative: bool
    rng: np.random.Generator
    mps_indices: Vector | None
    black_box_evals: np.ndarray | None
    norm: float

    def __init__(self, cross_strategy: CrossStrategy):
        self.error_norm = cross_strategy.error_norm
        self.num_samples = cross_strategy.num_samples
        self.error_relative = cross_strategy.error_relative
        self.rng = cross_strategy.rng
        # Cache
        self.mps_indices = None
        self.black_box_evals = None
        self.norm = 1.0

    def lp_distance(self, x: Vector) -> float:
        p = self.error_norm
        if np.isfinite(p):
            dist = ((1 / len(x)) * np.sum(np.abs(x) ** p)) ** (1 / p)
        else:
            dist = np.max(np.abs(x))
        return float(dist)

    def sample_error(self, cross: CrossInterpolation) -> float:
        if self.mps_indices is None:
            # Consider the allowed indices to impose restrictions (e.g. diagonal MPO)
            allowed_indices = getattr(cross.black_box, "allowed_indices", None)
            self.mps_indices = random_mps_indices(
                cross.black_box.physical_dimensions,
                self.num_samples,
                allowed_indices,
                self.rng,
            )
            self.black_box_evals = cross.sample(self.mps_indices).reshape(-1)
            self.norm = self.lp_distance(self.black_box_evals)
        mps_evals = evaluate_mps(cross.mps, self.mps_indices)
        error = self.lp_distance(mps_evals - self.black_box_evals)
        return error / self.norm if self.error_relative and self.norm else error


def check_tci_convergence(
    logger: Logger,
    half_sweep: int,
    results: CrossResults,
    cross_strategy: CrossStrategy,
) -> bool:
    """Checks the convergence of TCI from its trajectories and logs the results for each iteration."""
    iter_min, iter_max = cross_strategy.range_iters
    bond_min, bond_max = cross_strategy.range_max_bonds
    last = results.get_result(-1)
    maxbond = last.max_bond_dimension
    maxbond_prev = results.get_result(-2).max_bond_dimension if half_sweep > 2 else 0
    evals = last.evaluations
    error = last.error

    if logger:
        logger(
            f"Iteration (half-sweep): {half_sweep:3}/{iter_max}, "
            + f"error: {error:1.15e}/{cross_strategy.tol:.2e}, "
            + f"maxbond: {maxbond:3}/{bond_max}, "
            + f"evals: {evals:8}/{cross_strategy.max_evals}."
        )

    if half_sweep < iter_min or maxbond < bond_min:
        return False
    if error <= cross_strategy.tol:
        logger(f"State converged within tolerance {cross_strategy.tol}")
        return True
    elif maxbond - maxbond_prev <= 0:
        logger(f"Max. bond dimension converged with value {maxbond}")
        return True
    elif half_sweep >= iter_max:
        logger(f"Max. iterations reached at {iter_max}")
        return True
    elif maxbond >= bond_max:
        logger(f"Max. bond reached above the threshold {bond_max}")
        return True
    elif cross_strategy.max_time is not None and last.time >= cross_strategy.max_time:
        logger(f"Max. time reached above the threshold {cross_strategy.max_time}")
        return True
    elif cross_strategy.max_evals is not None and evals >= cross_strategy.max_evals:
        logger(f"Max. evals reached above the threshold {cross_strategy.max_evals}")
        return True

    return False


def cross_interpolation(
    cross_strategy: CrossStrategy,
    black_box: BlackBox,
    initial_points: Matrix | None = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using different the tensor
    cross-approximation (TCI) algorithm
    The black-box function can represent several different data structures.
    See `black_box` for usage examples.

    Parameters
    ----------
    cross_strategy : CrossStrategy
        A dataclass containing the parameters of the algorithm.
        See :class:`CrossStrategy`.
    black_box : BlackBox
        The black box to approximate as a MPS.
    initial_points : Matrix | None, default=None
        A collection of initial points used to initialize the algorithm.
        If None, an initial point at the origin is used.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    cross = cross_strategy.make_interpolator(black_box, initial_points)
    error_calculator = CrossError(cross_strategy)

    converged = False
    with make_logger(1) as logger:
        results = CrossResults(cross.mps)
        for half_sweep in range(1, cross_strategy.range_iters[1] + 1):
            forward = half_sweep % 2 == 1
            sites = (
                cross.iteration_range if forward else reversed(cross.iteration_range)
            )
            for k in sites:
                cross.update(k, forward)

            results.update(
                cross.mps,
                error_calculator.sample_error(cross),
                cross.mps.bond_dimensions(),
                cross.evals,
            )
            if not cross.two_sweeps_required or not forward:
                if converged := check_tci_convergence(
                    logger, half_sweep, results, cross_strategy
                ):
                    break

        if not converged:
            logger("Maximum number of iterations reached")

    return results
