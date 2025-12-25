from __future__ import annotations
import numpy as np
from ..typing import Tensor3, Tensor4, DenseOperator
from ..state import MPS
from ..cython import _contract_last_and_first
from ..state.environments import (
    _begin_environment,
    _update_right_environment,
    _update_left_environment,
)


class AntilinearForm:
    """Representation of a scalar product :math:`\\langle\\xi|\\psi\\rangle`
    with capabilities for differentiation.

    This class is an object that formally implements
    :math:`\\langle\\xi|\\psi\\rangle` with an argument :math:`\\xi`
    that may change and be updated over time. In particular, given a site `n`
    it can construct the tensor `L` such that the contraction between `L`
    and the `n`-th tensor from MPS :math:`\\xi` is the result of the linear form.

    Parameters
    ----------
    bra, ket: MPS
        MPS states :math:`\\xi` and :math:`\\psi` above.
    center: int, default = 0
        Position at which the `L` tensor is precomputed.
    """

    bra: MPS
    ket: MPS
    size: int
    R: list[DenseOperator]
    L: list[DenseOperator]
    center: int

    def __init__(self, bra: MPS, ket: MPS, center: int = 0):
        assert bra.size == ket.size
        size = bra.size
        ρ = _begin_environment()
        R = [ρ] * size
        for i in range(size - 1, center, -1):
            R[i - 1] = ρ = _update_right_environment(bra[i], ket[i], ρ)

        ρ = _begin_environment()
        L = [ρ] * size
        for i in range(0, center):
            L[i + 1] = ρ = _update_left_environment(bra[i], ket[i], ρ)

        self.bra = bra
        self.ket = ket
        self.size = size
        self.R = R
        self.L = L
        self.center = center

    def tensor1site(self) -> Tensor3:
        """Return the tensor representing the AntilinearForm at the
        `self.center` site."""
        center = self.center
        L = self.L[center]
        R = self.R[center]
        C = self.ket[center]
        return np.einsum("li,ijk,kn->ljn", L, C, R)

    def tensor2site(self, direction: int) -> Tensor4:
        """Return the tensor that represents the LinearForm using 'center'
        and another site.

        Parameters
        ----------
        direction : {+1, -1}
            If positive, the tensor acts on `self.center` and `self.center+1`
            Otherwise on `self.center` and `self.center-1`.

        Returns
        -------
        Tensor4
            Four-legged tensor representing the antilinear form.
        """
        if direction > 0:
            i = self.center
            j = i + 1
        else:
            j = self.center
            i = j - 1
        L = self.L[i]
        A = self.ket[i]
        B = self.ket[j]
        R = self.R[j]
        # np.einsum("li,ijk->ljk", L, A)
        LA = _contract_last_and_first(L, A)
        # np.einsum("kmn,no->kmo", B, R)
        BR = np.matmul(B, R)
        # np.einsum("ljk,kmo->ljmo", LA, BR)
        return _contract_last_and_first(LA, BR)

    def update(self, direction: int) -> None:
        """Notify that the `bra` state has been changed, and that we move to
        `self.center + direction`.

        We have updated 'mps' (the bra), which is now centered on a different point.
        We have to recompute the environments.

        Parameters
        ----------
        direction : { +1 , -1 }
        """
        if direction > 0:
            self.update_right()
        else:
            self.update_left()

    def update_right(self) -> None:
        """Notify that the `bra` state has been changed, and that we move to
        `self.center + 1`.

        We have updated 'mps' (the bra), which is now centered on a different point.
        We have to recompute the environments.
        """
        prev = self.center
        nxt = prev + 1
        assert nxt < self.size
        self.L[nxt] = _update_left_environment(
            self.bra[prev], self.ket[prev], self.L[prev]
        )
        self.center = nxt

    def update_left(self) -> None:
        """Notify that the `bra` state has been changed, and that we move to
        `self.center - 1`.

        We have updated 'mps' (the bra), which is now centered on a different point.
        We have to recompute the environments.
        """
        prev = self.center
        nxt = prev - 1
        assert nxt >= 0
        self.R[nxt] = _update_right_environment(
            self.bra[prev], self.ket[prev], self.R[prev]
        )
        self.center = nxt
