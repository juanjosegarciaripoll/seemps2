from __future__ import annotations
import numpy as np
import scipy.linalg
import functools
from typing import TypeAlias

from seemps.state import MPS, Strategy, DEFAULT_STRATEGY
from seemps.state.schmidt import _destructive_svd
from seemps.cython import destructively_truncate_vector, _contract_last_and_first
from seemps.analysis.mesh import Mesh, mesh_to_mps_indices
from seemps.analysis.cross import BlackBoxLoadMPS
from seemps.typing import Vector, Matrix, Tensor3
from ..tools import DEFAULT_RNG

IndexMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer]]


class SketchedCross:
    """
    Helper class for TT-RSS algorithm. Holds function multi-indices and enables sampling
    function fibers, similarly to TCI implementations.

    Parameters
    ----------
    black_box : BlackBoxLoadMPS
        Callable object that evaluates a function f on an array of MPS multi-indices
        considering the MPS quantization and structure given by the `map_matrix` and
        `physical_dimensions` attributes.
    samples : Matrix
        Array of pivot samples x_i ∈ ℝ^m, shape (D, m). These are quantized to indices
        and used to form the recursive prefix/suffix multi-index sets.
    """

    def __init__(self, black_box: BlackBoxLoadMPS, samples: Matrix):
        self.black_box = black_box
        self.sites = len(black_box.physical_dimensions)
        n = self.sites

        mesh_indices = _samples_to_mesh_indices(samples, black_box.mesh)
        if black_box.map_matrix is None:
            mps_indices = mesh_indices
        else:
            mps_indices = mesh_to_mps_indices(mesh_indices, black_box.map_matrix)

        # Sets of multi-index sets: left (I_l), physical (I_s) and right (I_r).
        # Equivalent to the recursive prefix and suffix sets S_k and T_k.
        self.I_l = [np.unique(mps_indices[:, :ℓ], axis=0) for ℓ in range(n + 1)]  # noqa: E741
        self.I_s = [np.arange(s).reshape(-1, 1) for s in black_box.physical_dimensions]
        self.I_r = [np.unique(mps_indices[:, ℓ:], axis=0) for ℓ in range(n + 1)]  # noqa: E741

    @staticmethod
    def combine_indices(*indices: IndexMatrix) -> IndexMatrix:
        def cartesian_column(A: Matrix, B: Matrix) -> Matrix:
            A_repeated = np.repeat(A, repeats=B.shape[0], axis=0)
            B_tiled = np.tile(B, (A.shape[0], 1))
            return np.hstack((A_repeated, B_tiled))

        return functools.reduce(cartesian_column, indices)

    def sample_fiber(self, k: int) -> Tensor3:
        i_l, i_s, i_r = self.I_l[k], self.I_s[k], self.I_r[k + 1]
        mps_indices = self.combine_indices(i_l, i_s, i_r)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_r)))

    @property
    def recursive_sets(self) -> list[tuple[Vector, Vector]]:
        β_list: list[Vector] = [np.zeros(self.I_l[1].shape[0], dtype=int)]
        x_list: list[Vector] = [self.I_l[1][:, 0]]
        for k in range(1, self.sites):
            S_k = self.I_l[k]
            S_kp1 = self.I_l[k + 1]
            index_map = {tuple(row): idx for idx, row in enumerate(S_k)}
            β_k = np.array([index_map[tuple(row[:-1])] for row in S_kp1], dtype=int)
            x_k = S_kp1[:, -1].astype(int)
            β_list.append(β_k)
            x_list.append(x_k)
        return list(zip(β_list, x_list))


def tt_rss(
    black_box: BlackBoxLoadMPS,
    samples: Matrix,
    max_bond_dimensions: Vector | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """
    Tensor Train via Recursive Sketching from Samples (TT-RSS).

    Implements the TT-RSS algorithm that computes the MPS representation of a black-box
    function from a set of pivot samples defining the region of interest through randomized
    recursive sketching.
    Source: https://arxiv.org/abs/2501.06300v1

    Parameters
    ----------
    black_box : BlackBoxLoadMPS
        A real-valued black-box function f(x_1,...,x_m), together with quantization
        parameters `map_matrix` and `physical_dimensions` representing MPS structure.
        Evaluates arrays of MPS indices, of shape (D, n), returning a vector of size D.
    samples : Matrix
        Array of shape (D, m), containing D samples xᵢ ∈ ℝ^m of the function.
        These are automatically quantized from m dimensions to n bits, to match the MPS
        dimensions defined on `black_box`.
    max_bond_dimensions : Vector, optional
        Vector of maximal bond dimensions `χ_k` allowed during the sketching procedure.
        If given, random Haar unitaries of size (t, t) are replaced by random isometries
        of size (t, χ_k), enhancing efficiency.
    strategy : Strategy, optional
        SVD rank-revealing strategy, determining the complexity of the target MPS.
        Defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        The MPS representation encoding the approximation of the target function.
    """
    # Sketch-forming
    sketched_cross = SketchedCross(black_box, samples)

    # Sketching
    Φ_tensors: list[Tensor3] = []
    for k in range(sketched_cross.sites):
        f_k = sketched_cross.sample_fiber(k)
        t = f_k.shape[2]
        if max_bond_dimensions is not None:
            χ = min(t, max_bond_dimensions[k])
        else:
            χ = min(t, strategy.get_max_bond_dimension())
        Φ_tensors.append(_contract_last_and_first(f_k, _random_isometry(t, χ)))

    # Trimming
    B_tensors: list[Tensor3] = []
    for Φ in Φ_tensors:
        _, d, t = Φ.shape
        U, S, _ = _destructive_svd(Φ.reshape(-1, t))
        destructively_truncate_vector(S, strategy)
        r = S.size
        B = U[:, :r] @ np.diag(S)
        B_tensors.append(B.reshape(-1, d, r))

    # System-forming
    s_sets = sketched_cross.recursive_sets
    A_matrices: list[Matrix] = []
    for k in range(sketched_cross.sites):
        β_k, x_k = s_sets[k]
        A_matrices.append(B_tensors[k][β_k, x_k, :])

    # Solving
    # TODO: fix global gauge/sign ambiguity
    cores: list[Tensor3] = [-B_tensors[0].copy()]  # Hardcoded minus sign
    for k, B_k in enumerate(B_tensors[1:], start=1):
        _, d_k, r_R = B_k.shape
        X_k, *_ = scipy.linalg.lstsq(
            A_matrices[k - 1],
            B_k.reshape(-1, d_k * r_R),
            cond=None,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=False,
            lapack_driver="gelsd",
        )
        cores.append(X_k.reshape(-1, d_k, r_R))

    return MPS(cores)


def _samples_to_mesh_indices(samples: Matrix, mesh: Mesh) -> Matrix:
    """
    Project continuous sample points onto the nearest nodes of a discretization mesh.

    Given a collection of continuous samples of a function defined over a discretization mesh,
    this routine maps each sample to the index of the nearest mesh point along each dimension.
    The mapping is performed by normalizing the samples to the mesh intervals and rounding to
    the closest grid node. The original sample locations are modified, resulting in a discretized
    approximation.
    """
    samples = np.asarray(samples, dtype=float)
    K, m = samples.shape
    if mesh.dimension != m:
        raise ValueError("Invalid dimensions.")

    indices = np.zeros((K, m), dtype=int)
    for dim, interval in enumerate(mesh.intervals):
        a, b, N = interval.start, interval.stop, interval.size
        samples_norm = (samples[:, dim] - a) / (b - a)
        indices[:, dim] = np.clip(
            np.round(samples_norm * (N - 1)).astype(int), 0, N - 1
        )
    return indices


def _random_isometry(
    rows: int, cols: int, rng: np.random.Generator = DEFAULT_RNG
) -> Matrix:
    if cols > rows:
        raise ValueError("cols must be <= rows")
    A = rng.normal(size=(rows, cols))
    Q, _ = scipy.linalg.qr(A, mode="economic", overwrite_a=True, check_finite=False)
    return Q


__all__ = ["tt_rss"]
