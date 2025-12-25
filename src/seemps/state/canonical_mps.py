from __future__ import annotations
import warnings
import numpy as np
from collections.abc import Sequence, Iterable
from ..typing import Vector, Tensor3, Tensor4, VectorLike, Environment
from .schmidt import (
    _vector2mps,
    _schmidt_weights,
    _left_orth_2site,
    _right_orth_2site,
)
from .environments import (
    _begin_environment,
    _update_left_environment,
    _update_right_environment,
)
from ..cython.core import (
    DEFAULT_STRATEGY,
    Strategy,
    _update_in_canonical_form_right,
    _update_in_canonical_form_left,
    _canonicalize,
    _recanonicalize,
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
    strategy: Strategy
    _error: float  # inherited, but Pyright wants us to confirm the type

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(
        self,
        data: Iterable[Tensor3],
        center: int | None = None,
        normalize: bool | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
        is_canonical: bool = False,
        error: float = 0,
    ):
        super().__init__(data, error)
        actual_center: int
        self.strategy = strategy
        if isinstance(data, CanonicalMPS):
            actual_center = self.center = data.center
            self._error = data._error
            if center is not None:
                actual_center = center
                self.recenter(actual_center)
        else:
            self.center = actual_center = self._interpret_center(
                0 if center is None else center
            )
            if not is_canonical:
                self._error += _canonicalize(self._data, actual_center, self.strategy)
        if normalize is True or (
            normalize is None and self.strategy.get_normalize_flag()
        ):
            A = self[actual_center]
            N = np.linalg.norm(A.reshape(-1))
            if N:
                self[actual_center] = A / N
            else:
                warnings.warn("Refusing to noramlize zero vector")

    @classmethod
    def from_vector(
        cls,
        ψ: VectorLike,
        dimensions: Sequence[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        center: int = 0,
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
        data, error = _vector2mps(ψ, dimensions, strategy, normalize, center)
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
        A: np.ndarray = self._data[self.center]
        # TODO: Find out why NumPy thinks np.vdot is of type bool
        return np.vdot(A, A).real  # pyright: ignore[reportReturnType]

    def left_environment(self, site: int) -> Environment:
        """Optimized version of :py:meth:`~seemps.state.MPS.left_environment`"""
        start = min(site, self.center)
        ρ = _begin_environment(self[start].shape[0])
        for A in self._data[start:site]:
            ρ = _update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site: int) -> Environment:
        """Optimized version of :py:meth:`~seemps.state.MPS.right_environment`"""
        start = max(site, self.center)
        ρ = _begin_environment(self[start].shape[-1])
        for A in self._data[start:site:-1]:
            ρ = _update_right_environment(A, A, ρ)
        return ρ

    def Schmidt_weights(self, site: int | None = None) -> Vector:
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
        return _schmidt_weights(self._data[site])

    def entanglement_entropy(self, site: int | None = None) -> float:
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

    def Renyi_entropy(self, site: int | None = None, alpha: float = 2.0) -> float:
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
    ) -> None:
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
            self.center, error = _update_in_canonical_form_right(
                self._data, A, self.center, truncation
            )
        else:
            self.center, error = _update_in_canonical_form_left(
                self._data, A, self.center, truncation
            )
        self._error += error

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
        self._data[site], self._data[site + 1], error = _left_orth_2site(AA, strategy)
        self.center = site + 1
        self._error += error

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
        self._data[site], self._data[site + 1], error = _right_orth_2site(AA, strategy)
        self.center = site
        self._error += error

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

    def recenter(self, center: int, strategy: Strategy | None = None) -> CanonicalMPS:
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
        newcenter = self._interpret_center(center)
        oldcenter = self.center
        if newcenter != oldcenter:
            self._error += _recanonicalize(
                self._data,
                oldcenter,
                newcenter,
                self.strategy if strategy is None else strategy,
            )
            self.center = newcenter
        return self

    def normalize_inplace(self) -> CanonicalMPS:
        """Normalize the state by updating the central tensor."""
        n = self.center
        A = self._data[n]
        N = np.linalg.norm(A.reshape(-1))
        if N:
            self._data[n] = A / N
        return self

    def __copy__(self):
        """Return a shallow copy of the CanonicalMPS, preserving the tensors."""
        return type(self)(
            self, center=self.center, strategy=self.strategy, error=self._error
        )

    def copy(self):
        """Return a shallow copy of the CanonicalMPS, preserving the tensors."""
        return self.__copy__()


__all__ = ["CanonicalMPS"]
