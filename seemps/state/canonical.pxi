import numpy as np
from ..typing import *
from . import schmidt

# TODO: Replace einsum by a more efficient form
def _update_in_canonical_form_right(
    Ψ: list[Tensor3], A: Tensor3, site: int, truncation: Strategy
) -> tuple[int, float]:
    """Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process."""
    if site + 1 == len(Ψ):
        Ψ[site] = A
        return site, 0.0
    Ψ[site], sV, err = schmidt.ortho_right(A, truncation)
    site += 1
    # np.einsum("ab,bic->aic", sV, Ψ[site])
    Ψ[site] = _contract_last_and_first(sV, Ψ[site])
    return site, err


# TODO: Replace einsum by a more efficient form
def _update_in_canonical_form_left(
    Ψ: list[Tensor3], A: Tensor3, site: int, truncation: Strategy
) -> tuple[int, float]:
    """Insert a tensor in canonical form into the MPS Ψ at the given site.
    Update the neighboring sites in the process."""
    if site == 0:
        Ψ[site] = A
        return site, 0.0
    Ψ[site], Us, err = schmidt.ortho_left(A, truncation)
    site -= 1
    # np.einsum("aib,bc->aic", Ψ[site], Us)
    Ψ[site] = np.matmul(Ψ[site], Us)
    return site, err


def _canonicalize(Ψ: list[Tensor3], center: int, truncation: Strategy) -> float:
    """Update a list of `Tensor3` objects to be in canonical form
    with respect to `center`."""
    # TODO: Revise the cumulative error update. Does it follow update_error()?
    err = 0.0
    for i in range(0, center):
        _, errk = _update_in_canonical_form_right(Ψ, Ψ[i], i, truncation)
        err += errk
    for i in range(len(Ψ) - 1, center, -1):
        _, errk = _update_in_canonical_form_left(Ψ, Ψ[i], i, truncation)
        err += errk
    return err


cdef class CanonicalMPS(MPS):
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
        Defaults to `False`.
    strategy : Strategy, optional
        The truncation strategy for the orthogonalization and later
        algorithms. Defaults to `DEFAULT_STRATEGY`.
    """
    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    def __init__(
        self,
        data: Iterable[Tensor3],
        center: Optional[int] = None,
        normalize: bool = False,
        strategy: Strategy = DEFAULT_STRATEGY,
        **kwdargs,
    ):
        cdef CanonicalMPS cdata

        super().__init__(data, **kwdargs)
        actual_center: int
        self._strategy = strategy
        if isinstance(data, CanonicalMPS):
            cdata = data
            actual_center = self._center = cdata._center
            self._error = cdata._error
            if center is not None:
                actual_center = center
                self.recenter(actual_center)
        else:
            self._center = actual_center = self._interpret_center(
                0 if center is None else center
            )
            self.update_error(_canonicalize(self._data, actual_center, self._strategy))
        if normalize or self._strategy.get_normalize_flag():
            A = self[actual_center]
            self[actual_center] = A / np.linalg.norm(A)

    @classmethod
    def from_vector(
        cls,
        ψ: VectorLike,
        dimensions: Sequence[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
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

        Returns
        -------
        CanonicalMPS
            A valid matrix-product state approximating this state vector.

        See also
        --------
        :py:meth:`~seemps.state.MPS.from_vector`
        """
        return CanonicalMPS(
            schmidt.vector2mps(ψ, dimensions, strategy, normalize),
            center=kwdargs.get("center", 0),
            strategy=strategy,
        )

    def norm_squared(self) -> float:
        """Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        A = self._data[self._center]
        return np.vdot(A, A).real

    def left_environment(self, site: int) -> Environment:
        """Optimized version of :py:meth:`~seemps.state.MPS.left_environment`"""
        cdef Py_ssize_t k, start = min(site, self._center)
        rho = begin_environment_with_D(self._data[start].shape[0])
        for k in range(start, site):
            A = self._data[k]
            rho = update_left_environment(A, A, rho, None)
        return rho

    def right_environment(self, site: int) -> Environment:
        """Optimized version of :py:meth:`~seemps.state.MPS.right_environment`"""
        cdef Py_ssize_t k, start = max(site, self._center)
        rho = begin_environment_with_D(self._data[start].shape[2])
        for k in range(start, site, -1):
            A = self._data[k]
            rho = update_right_environment(A, A, rho, None)
        return rho

    def entanglement_entropy(self, site: Optional[int] = None) -> float:
        """Compute the entanglement entropy of the MPS for a bipartition
        around `site`.

        Parameters
        ----------
        site : int, optional
            Site in the range `[0, self.size)`, defaulting to `self._center`.
            The system is diveded into `[0, self.site)` and `[self.site, self.size)`.

        Returns
        -------
        float
            Von Neumann entropy of bipartition.
        """
        if site is None:
            site = self._center
        if site != self._center:
            return self.copy().recenter(site).entanglement_entropy()
        # TODO: this is for [0, self._center] (self._center, self.size)
        # bipartitions, but we can also optimizze [0, self._center) [self._center, self.size)
        A = self._data[site]
        d1, d2, d3 = A.shape
        s = schmidt.svd(
            A.reshape(d1 * d2, d3),
            full_matrices=False,
            compute_uv=False,
            check_finite=False,
            lapack_driver=schmidt.SVD_LAPACK_DRIVER,
        )
        return -np.sum(2 * s * s * np.log2(s))

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
            self._center, err = _update_in_canonical_form_right(
                self._data, A, self._center, truncation
            )
        else:
            self._center, err = _update_in_canonical_form_left(
                self._data, A, self._center, truncation
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
        self._data[site], self._data[site + 1], err = schmidt.left_orth_2site(
            AA, strategy
        )
        self._center = site + 1
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
        self._data[site], self._data[site + 1], err = schmidt.right_orth_2site(
            AA, strategy
        )
        self._center = site
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
        old = self._center
        if strategy is None:
            strategy = self._strategy
        if center != old:
            dr = +1 if center > old else -1
            for i in range(old, center, dr):
                self.update_canonical(self._data[i], dr, strategy)
        return self

    def normalize_inplace(self) -> CanonicalMPS:
        """Normalize the state by updating the central tensor."""
        n = self._center
        A = self._data[n]
        self._data[n] = A / np.linalg.norm(A)
        return self

    def copy(self):
        """Return a shallow copy of the CanonicalMPS, preserving the tensors."""
        cdef CanonicalMPS output = CanonicalMPS.__new__(CanonicalMPS)
        output._data = list(self._data)
        output._size = self._size
        output._error = self._error
        output._center = self._center
        output._strategy = self._strategy
        return output

    @property
    def strategy(self):
        return self._strategy

    @property
    def center(self):
        return self._center
