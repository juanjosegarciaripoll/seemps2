from __future__ import annotations
import warnings
import numpy as np
from typing import Optional, Sequence, Iterable
from ..typing import Vector, Tensor3, Tensor4, VectorLike, Environment
from . import environments, schmidt
from seemps.state.core import (
    DEFAULT_STRATEGY,
    Strategy,
    _update_canonical_right,
    _update_canonical_left,
    _update_canonical_2site_left,
    _update_canonical_2site_right,
    _canonicalize,
)
from .mps import MPS


class CanonicalMPS(MPS):
    """Canonical MPS class.

    This implements a Matrix Product State object with open boundary
    conditions, that is always on canonical form with respect to a given site.
    The tensors have three indices, `A[α,i,β]`, where `α,β` are the internal
    labels and `i` is the physical state of the given site.

    Parameters
    ----------
    data : Iterable[Tensor3]
        A set of tensors that will be orthogonalized. It can be an
        :class:`MPS` state.
    center : int, optional
        The center for the canonical form. Defaults to the first site
        `center = 0`.
    normalize : bool, optional
        Whether to normalize the state to compensate for truncation errors.
        Defaults to the value set by `strategy`.
    strategy : Strategy, optional
        The truncation strategy for the orthogonalization and later
        algorithms. Defaults to `DEFAULT_STRATEGY`.
    """

    center: int

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(
        self,
        data: Iterable[Tensor3],
        center: Optional[int] = None,
        normalize: Optional[bool] = None,
        strategy: Strategy = DEFAULT_STRATEGY,
        is_canonical: bool = False,
        **kwdargs,
    ):
        actual_center: int
        self.strategy = strategy
        if isinstance(data, CanonicalMPS):
            super().__init__(data._data.copy(), **kwdargs)
            actual_center = self.center = data.center
            self._error = data._error
            if center is not None:
                actual_center = center
                self.recenter(actual_center)
        else:
            super().__init__(data, **kwdargs)
            self.center = actual_center = self._interpret_center(
                0 if center is None else center
            )
            if not is_canonical:
                self.update_error(_canonicalize(self._data, actual_center, strategy))
        if normalize is True or (normalize is None and strategy.get_normalize_flag()):
            A = self[actual_center]
            N = np.linalg.norm(A)
            if N:
                self[actual_center] = A / np.linalg.norm(A)

    @classmethod
    def from_vector(
        cls,
        ψ: VectorLike,
        dimensions: Sequence[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        center: int = 0,
        **kwdargs,
    ) -> CanonicalMPS:
        """Create an MPS in canonical form starting from a state vector.

        Parameters
        ----------
        ψ : VectorLike
            Real or complex vector of a wavefunction.
        dimensions : Sequence[int]
            Sequence of integers representing the dimensions of the
            quantum systems that form this state.
        strategy : Strategy, default = DEFAULT_STRATEGY
            Default truncation strategy for algorithms working on this state.
        normalize : bool, default = True
            Whether the state is normalized to compensate truncation errors.
        center : int, default = 0
            Center for the canonical form of this decomposition.

        Returns
        -------
        CanonicalMPS
            A valid matrix-product state approximating this state vector.

        See also
        --------
        :py:meth:`~seemps.state.MPS.from_vector`
        """
        data, error = schmidt.vector2mps(ψ, dimensions, strategy, normalize, center)
        return CanonicalMPS(
            data,
            error=error,
            center=center,
            is_canonical=True,
        )

    def zero_state(self) -> CanonicalMPS:
        """Return a zero wavefunction with the same physical dimensions."""
        return CanonicalMPS(
            [np.zeros((1, A.shape[1], 1)) for A in self._data],
            error=0.0,
            center=0,
            is_canonical=True,
        )

    def norm_squared(self) -> float:
        """Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        A = self._data[self.center]
        return np.vdot(A, A).real

    def left_environment(self, site: int) -> Environment:
        """Optimized version of :py:meth:`~seemps.state.MPS.left_environment`"""
        start = min(site, self.center)
        ρ = environments.begin_environment(self[start].shape[0])
        for A in self._data[start:site]:
            ρ = environments.update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site: int) -> Environment:
        """Optimized version of :py:meth:`~seemps.state.MPS.right_environment`"""
        start = max(site, self.center)
        ρ = environments.begin_environment(self[start].shape[-1])
        for A in self._data[start:site:-1]:
            ρ = environments.update_right_environment(A, A, ρ)
        return ρ

    def Schmidt_weights(self, site: Optional[int] = None) -> Vector:
        """Return the Schmidt weights for a bipartition around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self.center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.

        Returns
        -------
        numbers: np.ndarray
            Vector of non-negative Schmidt weights.
        """
        if site is None:
            site = self.center
        else:
            site = self._interpret_center(site)
        if site != self.center:
            return self.copy().recenter(site).Schmidt_weights()
        # TODO: this is for [0, self.center] (self.center, self.size)
        # bipartitions, but we can also optimizze [0, self.center) [self.center, self.size)
        return schmidt.schmidt_weights(self._data[site])

    def entanglement_entropy(self, site: Optional[int] = None) -> float:
        """Compute the entanglement entropy of the MPS for a bipartition
        around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self.center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.

        Returns
        -------
        float
            Von Neumann entropy of bipartition.
        """
        s = self.Schmidt_weights(site)
        return -np.sum(s * np.log2(s))

    def Renyi_entropy(self, site: Optional[int] = None, alpha: float = 2.0) -> float:
        """Compute the Renyi entropy of the MPS for a bipartition
        around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self.center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.
        alpha : float, default = 2
            Power of the Renyi entropy.

        Returns
        -------
        float
            Von Neumann entropy of bipartition.
        """
        s = self.Schmidt_weights(site)
        if alpha < 0:
            raise ValueError("Invalid Renyi entropy power")
        if alpha == 0:
            alpha = 1e-9
        elif alpha == 1:
            alpha = 1 - 1e-9
        S = np.log(np.sum(s**alpha)) / (1 - alpha)
        return S

    def update_canonical(
        self, A: Tensor3, direction: int, truncation: Strategy
    ) -> float:
        """Update the state, replacing the tensor at `self.center`
        and moving the center to `self.center + direction`.

        Parameters
        ----------
        A : Tensor3
            The new tensor.
        direction : { +1, -1 }
            Direction in which the update is performed.
        truncation : Strategy
            Truncation parameters such as tolerance or maximum
            bond dimension.

        Returns
        -------
        float
            The truncation error of this update.
        """
        if direction > 0:
            self.center, err = _update_canonical_right(
                self._data, A, self.center, truncation
            )
        else:
            self.center, err = _update_canonical_left(
                self._data, A, self.center, truncation
            )
        self.update_error(err)
        return err

    # TODO: check if `site` is not needed, as it should be self.center
    def update_2site_right(self, AA: Tensor4, site: int, strategy: Strategy) -> None:
        """Split a two-site tensor into two one-site tensors by
        right orthonormalization and insert the tensor in canonical form into
        the MPS at the given site and the site on its right. Update the
        neighboring sites in the process.

        Parameters
        ----------
        AA : Tensor4
            Two-site tensor `A[a,i,j,b]`
        site : int
            The index of the site whose quantum number is `i`. The new center
            will be `self.site+1`.
        strategy : Strategy
            Truncation strategy, including relative tolerances and maximum
            bond dimensions
        """
        err = _update_canonical_2site_left(self._data, AA, site, strategy)
        self.center = site + 1
        self.update_error(err)

    def update_2site_left(self, AA: Tensor4, site: int, strategy: Strategy) -> None:
        """Split a two-site tensor into two one-site tensors by
        left orthonormalization and insert the tensor in canonical form into the
        MPS Ψ at the given site and the site on its right. Update the
        neighboring sites in the process.

        Parameters
        ----------
        AA : Tensor4
            Two-site tensor `A[a,i,j,b]`
        site : int
            The index of the site whose quantum number is `i`. The new center
            will be `self.site`.
        strategy : Strategy
            Truncation strategy, including relative tolerances and maximum
            bond dimensions
        """
        err = _update_canonical_2site_right(self._data, AA, site, strategy)
        self.center = site
        self.update_error(err)

    def _interpret_center(self, center: int) -> int:
        """Converts `center` into an integer in `[0,self.size)`, with the
        convention that `-1 = size-1`, `-2 = size-2`, etc. Trows an exception of
        `center` if out of bounds."""
        size = self.size
        if 0 <= center < size:
            return center
        center += size
        if 0 <= center < size:
            return center
        raise IndexError()

    def recenter(
        self, center: int, strategy: Optional[Strategy] = None
    ) -> CanonicalMPS:
        """Update destructively the state to be in canonical form with respect
        to a different site.

        Parameters
        ----------
        center : int
            The new site for orthogonalization in `[0, self.size)`
        strategy : Strategy, optional
            Truncation strategy. Defaults to `self.strategy`

        Returns
        -------
        CanonicalMPS
            This same object.
        """
        center = self._interpret_center(center)
        old = self.center
        if strategy is None:
            strategy = self.strategy
        if center != old:
            dr = +1 if center > old else -1
            for i in range(old, center, dr):
                self.update_canonical(self._data[i], dr, strategy)
        return self

    def normalize_inplace(self) -> CanonicalMPS:
        """Normalize the state by updating the central tensor."""
        n = self.center
        A = self._data[n]
        N = np.linalg.norm(A)
        if N == 0:
            warnings.warn("Refusing to normalize zero vector")
        else:
            self._data[n] = A / N
        return self

    def __copy__(self):
        """Return a shallow copy of the CanonicalMPS, preserving the tensors."""
        return type(self)(
            self, center=self.center, strategy=self.strategy, error=self.error
        )

    def copy(self):
        """Return a shallow copy of the CanonicalMPS, preserving the tensors."""
        return self.__copy__()
