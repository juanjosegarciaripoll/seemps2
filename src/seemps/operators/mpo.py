from __future__ import annotations
from collections.abc import Sequence
from typing import overload
import warnings
import numpy as np
from ..tools import InvalidOperation
from ..typing import Tensor4, Tensor3, DenseOperator, Weight
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, MPSSum, Strategy, TensorArray
from ..state.environments import (
    scprod,
    begin_mpo_environment,
    update_left_mpo_environment,
    update_right_mpo_environment,
    join_mpo_environments,
)


def _mpo_multiply_tensor(A: Tensor4, B: Tensor3):
    # Implements
    # np.einsum("cjd,aijb->caidb", B, A)
    #
    # Matmul takes two arguments
    #     B(c, 1, 1, d, j)
    #     A(1, a, i, j, b)
    # It broadcasts, repeating the indices that are of size 1
    #     B(c, a, i, d, j)
    #     A(c, a, i, j, b)
    # And then multiplies the matrices that are formed by the last two
    # indices, (d,j) * (j,b) -> (b,d) so that the outcome has size
    #     C(c, a, i, d, b)
    #
    a, i, j, b = A.shape
    c, j, d = B.shape
    # np.matmul(...) -> C(a,i,b,c,d)
    return np.matmul(
        B.transpose(0, 2, 1).reshape(c, 1, 1, d, j), A.reshape(1, a, i, j, b)
    ).reshape(c * a, i, d * b)


class MPO(TensorArray):
    """Matrix Product Operator class.

    This implements a bare-bones Matrix Product Operator object with open
    boundary conditions. The tensors have four indices, A[α,i,j,β], where
    'α,β' are the internal labels and 'i,j' the physical indices ar the given
    site.

    Parameters
    ----------
    data: Sequence[Tensor4]
        Sequence of four-legged tensors forming the structure.
    strategy: Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for algorithms.
    """

    strategy: Strategy

    __array_priority__: int = 10000

    def __init__(self, data: Sequence[Tensor4], strategy: Strategy = DEFAULT_STRATEGY):
        super().__init__(data)
        self.strategy = strategy

    def copy(self) -> MPO:
        """Return a shallow copy of the MPO, without duplicating the tensors."""
        # We use the fact that TensorArray duplicates the list
        return MPO(self, self.strategy)

    def __add__(self, A: MPO | MPOList | MPOSum) -> MPOSum:
        """Represent `self + A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, 1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + A.weights, A.strategy)
        raise TypeError(f"Cannod add MPO and {type(A)}")

    def __sub__(self, A: MPO | MPOList | MPOSum) -> MPOSum:
        """Represent `self - A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, -1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + [-w for w in A.weights], A.strategy)
        raise TypeError(f"Cannod subtract MPO and {type(A)}")

    def __mul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `self * n`"""
        if isinstance(n, (int, float, complex)):
            absn = abs(n)
            if absn:
                phase = n / absn
                factor = np.exp(np.log(absn) / self.size)
            else:
                phase = 0.0
                factor = 0.0
            return MPO(
                [
                    (factor if i > 0 else (factor * phase)) * A
                    for i, A in enumerate(self)
                ],
                self.strategy,
            )
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `n * self`"""
        if isinstance(n, (int, float, complex)):
            absn = abs(n)
            if absn:
                phase = n / absn
                factor = np.exp(np.log(absn) / self.size)
            else:
                phase = 0.0
                factor = 0.0
            return MPO(
                [
                    (factor if i > 0 else (factor * phase)) * A
                    for i, A in enumerate(self)
                ],
                self.strategy,
            )
        raise InvalidOperation("*", n, self)

    def __pow__(self, n: int) -> MPOList:
        """Exponentiate a MPO to n."""
        if isinstance(n, int):
            return MPOList([self.copy() for _ in range(n)])
        raise InvalidOperation("**", n, self)

    def dimensions(self) -> list[int]:
        """Return the physical dimensions (Deprecated, see :meth:`dimensions`)."""
        warnings.warn(
            "MPO*.dimensions is deprecated. Use MPO*.physical_dimensions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.physical_dimensions()

    def physical_dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPO."""
        return [A.shape[2] for A in self]

    def bond_dimensions(self) -> list[int]:
        """Return the bond dimensions of the MPO."""
        return [A.shape[-1] for A in self][:-1]

    def max_bond_dimension(self) -> int:
        """Return the largest bond dimension."""
        return max(A.shape[0] for A in self)

    @property
    def T(self) -> MPO:
        """Return the transpose of this operator."""
        return MPO([A.transpose(0, 2, 1, 3) for A in self], self.strategy)

    def tomatrix(self) -> DenseOperator:
        """Convert this MPO to a dense or sparse matrix (Deprecated, see :meth:`to_matrix`)."""
        warnings.warn(
            "MPO.tomatrix() has been renamed to_matrix()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_matrix()

    def to_matrix(self) -> DenseOperator:
        """Convert this MPO to a dense or sparse matrix."""
        Di = 1  # Total physical dimension so far
        Dj = 1
        out = np.array([[[1.0]]])
        for A in self:
            _, i, j, b = A.shape
            out = np.einsum("lma,aijb->limjb", out, A)
            Di *= i
            Dj *= j
            out = out.reshape(Di, Dj, b)
        return out[:, :, 0]

    def set_strategy(self, strategy: Strategy) -> MPO:
        """Return MPO with the given strategy."""
        return MPO(self, strategy)

    @overload
    def apply(
        self,
        state: MPS,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS: ...

    @overload
    def apply(
        self,
        state: MPSSum,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS: ...

    def apply(
        self,
        state: MPS | MPSSum,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS | MPSSum:
        """Implement multiplication `A @ state` between a matrix-product operator
        `A` and a matrix-product state `state`.

        Parameters
        ----------
        state : MPS | MPSSum
            Transformed state.
        strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY
        simplify : bool, optional
            Whether to simplify the state after the contraction.
            Defaults to `strategy.get_simplify_flag()`

        Returns
        -------
        CanonicalMPS
            The result of the contraction.
        """
        # TODO: Remove implicit conversion of MPSSum to MPS
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        if isinstance(state, MPSSum):
            assert self.size == state.size
            for i, (w, mps) in enumerate(zip(state.weights, state.states)):
                Ostate = w * MPS(
                    [_mpo_multiply_tensor(A, B) for A, B in zip(self, mps)],
                    error=mps.error(),
                )
                state = Ostate if i == 0 else state + Ostate
        elif isinstance(state, MPS):
            assert self.size == state.size
            state = MPS(
                [_mpo_multiply_tensor(A, B) for A, B in zip(self, state)],
                error=state.error(),
            )
        else:
            raise TypeError(f"Cannot multiply MPO with {state}")

        if simplify:
            state = simplify_mps(state, strategy=strategy)
        return state

    @overload
    def __matmul__(self, b: MPS) -> MPS: ...

    @overload
    def __matmul__(self, b: MPSSum) -> MPS | MPSSum: ...

    def __matmul__(self, b: MPS | MPSSum) -> MPS | MPSSum:
        """Implement multiplication `self @ b`."""
        return self.apply(b)

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Sequence[int] | None = None,
        dimensions: int | list[int] = 2,
    ) -> MPO:
        """Enlarge an MPO so that it acts on a larger Hilbert space with 'L' sites.

        Parameters
        ----------
        L : int
            The new size of the MPS. Must be strictly larger than `self.size`.
        sites : Iterable[int], optional
            Sequence of integers describing the sites that occupied by the
            tensors in this state.
        dimensions : int | list[int], default = 2
            Dimension of the added sites. It can be the same integer or a list
            of integers with the same length as `sites`.

        Returns
        -------
        MPO
            Extended MPO.
        """
        if isinstance(dimensions, int):
            final_dimensions = [dimensions] * max(L - self.size, 0)
        else:
            final_dimensions = dimensions.copy()
            assert len(dimensions) == L - self.size
        if sites is None:
            sites = range(self.size)
        assert L >= self.size
        assert len(sites) == self.size

        data: list[np.ndarray] = [np.ndarray(())] * L
        for ndx, A in zip(sites, self):
            data[ndx] = A
        D = 1
        k = 0
        for i, A in enumerate(data):
            if A.ndim == 0:
                d = final_dimensions[k]
                A = np.eye(D).reshape(D, 1, 1, D) * np.eye(d).reshape(1, d, d, 1)
                data[i] = A
                k = k + 1
            else:
                D = A.shape[-1]
        return MPO(data, strategy=self.strategy)

    def expectation(self, bra: MPS, ket: MPS | None = None) -> Weight:
        """Expectation value of MPO on one or two MPS states.

        If one state is given, this state is interpreted as :math:`\\psi`
        and this function computes :math:`\\langle{\\psi|O\\psi}\\rangle`
        If two states are given, the first one is the bra :math:`\\psi`,
        the second one is the ket :math:`\\phi`, and this computes
        :math:`\\langle\\psi|O|\\phi\\rangle`.

        Parameters
        ----------
        bra : MPS
            The state :math:`\\psi` on which the expectation value
            is computed.
        ket : MPS | None
            The ket component of the expectation value. Defaults to `bra`.

        Returns
        -------
        float | complex
            :math:`\\langle\\psi\\vert{O}\\vert\\phi\\rangle` where `O`
            is the matrix-product operator.
        """
        if isinstance(bra, CanonicalMPS):
            center = bra.center
        elif isinstance(bra, MPS):
            center = self.size - 1
        else:
            raise Exception("MPS required")
        if ket is None:
            ket = bra
        elif not isinstance(ket, MPS):
            raise Exception("MPS required")
        left = right = begin_mpo_environment()
        for i in range(0, center):
            left = update_left_mpo_environment(left, bra[i], self[i], ket[i])
        for i in range(self.size - 1, center - 1, -1):
            right = update_right_mpo_environment(right, bra[i], self[i], ket[i])
        return join_mpo_environments(left, right)

    def reverse(self) -> MPO:
        """Reverse the sites and tensors.

        Creates a new matrix product operator where tensors `0, 1, ..., N-1`
        are mapped to `N-1, N-2, ..., 0`. For the MPO to be consistent, this
        also implies reversing the order of the intermediate indices. Thus,
        if we label as `A` and `B` the tensors of the original and of the
        reversed MPOs, we have

        .. math::
            B_{a_{n-1},i_n,j_n,a_n} = A_{a_{N-n-1},i_{N-n-1},j_{N-n-1},a_{N-n-2}}
        """
        return MPO(
            [
                np.moveaxis(op, [0, 1, 2, 3], [3, 1, 2, 0])
                for op in reversed(self._data)
            ],
            self.strategy,
        )


class MPOList(object):
    """Sequence of matrix-product operators.

    This implements a list of MPOs that are applied sequentially. It can impose
    its own truncation or simplification strategy on top of the one provided by
    the individual operators.

    Parameters
    ----------
    mpos : list[MPO]
        Operators in this sequence, to be applied from mpos[0] to mpos[-1]. Must
        contain at least one operator.
    strategy : Strategy, optional
        Truncation and simplification strategy, defaults to DEFAULT_STRATEGY

    Attributes
    ----------
    mpos : list[MPO]
        Operators in this sequence, to be applied from mpos[0] to mpos[-1]. Must
        contain at least one operator.
    strategy : Strategy
        Truncation and simplification strategy.
    size : int
        Number of quantum subsystems in each MPO. Computed from the supplied
        MPOs. Not checked for consistency.
    """

    __array_priority__: int = 10000

    mpos: list[MPO]
    strategy: Strategy
    size: int

    def __init__(self, mpos: Sequence[MPO], strategy: Strategy = DEFAULT_STRATEGY):
        assert len(mpos) > 1
        self.mpos = mpos = list(mpos)
        self.size = mpos[0].size
        self.strategy = strategy

    def copy(self) -> MPOList:
        """Shallow copy of the MPOList, without copying the MPOs themselves."""
        return MPOList(self.mpos.copy(), self.strategy)

    def __add__(self, A: MPO | MPOList | MPOSum) -> MPOSum:
        """Represent `self + A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, 1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + A.weights, A.strategy)
        raise TypeError(f"Cannod add MPO and {type(A)}")

    def __sub__(self, A: MPO | MPOList | MPOSum) -> MPOSum:
        """Represent `self - A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, -1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + [-w for w in A.weights], A.strategy)
        raise TypeError(f"Cannod subtract MPO and {type(A)}")

    def __mul__(self, n: Weight) -> MPOList:
        """Multiply an MPO by a scalar `n` as in `n * self`."""
        if isinstance(n, (int, float, complex)):
            return MPOList([n * self.mpos[0]] + self.mpos[1:], self.strategy)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPOList:
        """Multiply an MPO by a scalar `n` as in `self * n`."""
        if isinstance(n, (int, float, complex)):
            return MPOList([n * self.mpos[0]] + self.mpos[1:], self.strategy)
        raise InvalidOperation("*", n, self)

    @property
    def T(self) -> MPOList:
        """Return the transpose of this operator."""
        return MPOList([A.T for A in reversed(self.mpos)], self.strategy)

    def dimensions(self) -> list[int]:
        """Return the physical dimensions (Deprecated, see :meth:`dimensions`)."""
        warnings.warn(
            "MPO*.dimensions is deprecated. Use MPO*.physical_dimensions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.physical_dimensions()

    def physical_dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPOList (Deprecated, see :meth:`dimensions`)."""
        return self.mpos[0].dimensions()

    def tomatrix(self) -> DenseOperator:
        """Convert this MPO to a dense or sparse matrix (Deprecated, see :meth:`to_matrix`)."""
        warnings.warn(
            "MPO.tomatrix() has been renamed to_matrix()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_matrix()

    def to_matrix(self) -> DenseOperator:
        """Convert this MPO to a dense or sparse matrix."""
        A = self.mpos[0].to_matrix()
        for mpo in self.mpos[1:]:
            A = mpo.to_matrix() @ A
        return A

    def set_strategy(
        self, strategy: Strategy, strategy_components: Strategy | None = None
    ) -> MPOList:
        """Return MPOList with the given strategy."""
        if strategy_components is not None:
            mpos = [mpo.set_strategy(strategy_components) for mpo in self.mpos]
        else:
            mpos = self.mpos
        return MPOList(mpos=mpos, strategy=strategy)

    @overload
    def apply(
        self,
        state: MPS,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS: ...

    @overload
    def apply(
        self,
        state: MPSSum,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS | MPSSum: ...

    # TODO: Describe how `strategy` and simplify act as compared to
    # the values provided by individual operators.
    def apply(
        self,
        state: MPS | MPSSum,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS | MPSSum:
        """Implement multiplication `A @ state` between a matrix-product operator
        `A` and a matrix-product state `state`.

        Parameters
        ----------
        state : MPS | MPSSum
            Transformed state.
        strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY
        simplify : bool, optional
            Whether to simplify the state after the contraction.
            Defaults to `strategy.get_simplify_flag()`

        Returns
        -------
        CanonicalMPS
            The result of the contraction.
        """
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()

        for mpo in self.mpos:
            # log(f'Total error before applying MPOList {b.error()}')
            state = mpo.apply(state)
        if simplify:
            state = simplify_mps(state, strategy=strategy)
        return state

    @overload
    def __matmul__(self, b: MPS) -> MPS: ...

    @overload
    def __matmul__(self, b: MPSSum) -> MPS | MPSSum: ...

    def __matmul__(self, b: MPS | MPSSum) -> MPS | MPSSum:
        """Implement multiplication `self @ b`."""
        return self.apply(b)

    def extend(
        self, L: int, sites: list[int] | None = None, dimensions: int | list[int] = 2
    ) -> MPOList:
        """Enlarge an MPOList so that it acts on a larger Hilbert space with 'L' sites.

        See also
        --------
        :py:meth:`MPO.extend`
        """
        return MPOList(
            [mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self.mpos],
            strategy=self.strategy,
        )

    def _joined_tensors(self, i: int, L: int) -> Tensor4:
        """Join the tensors from all MPOs into bigger tensors."""

        def join(A: Tensor4, *args: Tensor4) -> Tensor4:
            if not args:
                return A
            B = join(*args)
            a, d, d, b = A.shape
            c, d, d, e = B.shape
            # A, B, args[1],... are the tensors of the MPO to
            # join. They are applied to the MPS in this order, hence the
            # particular position of elements in opt_einsum
            # return opt_einsum.contract("aijb,cjkd->acikbd", B, A).reshape(
            #    a * c, d, d, b * e
            # )
            # aijbc,cjkd->aibckd->acikbd
            aux = np.tensordot(B, A, ((2,), (1,)))
            return np.ascontiguousarray(
                aux.transpose(0, 3, 1, 4, 2, 5).reshape(a * c, d, d, b * e)
            )

        return join(*[mpo[i] for mpo in self.mpos])

    def join(self, strategy: Strategy | None = None) -> MPO:
        """Create an `MPO` by combining all tensors from all MPOs.

        Returns
        -------
        MPO
            Quantum operator implementing the product of tensors.
        """
        L = self.mpos[0].size
        return MPO(
            [self._joined_tensors(i, L) for i in range(L)],
            strategy=self.strategy if strategy is None else strategy,
        )

    def expectation(self, bra: MPS, ket: MPS | None = None) -> Weight:
        """Expectation value of MPOList on one or two MPS states.

        If one state is given, this state is interpreted as :math:`\\psi`
        and this function computes :math:`\\langle{\\psi|O\\psi}\\rangle`
        If two states are given, the first one is the bra :math:`\\psi`,
        the second one is the ket :math:`\\phi`, and this computes
        :math:`\\langle\\psi|O|\\phi\\rangle`.

        Parameters
        ----------
        bra : MPS
            The state :math:`\\psi` on which the expectation value
            is computed.
        ket : MPS | None
            The ket component of the expectation value. Defaults to `bra`.

        Returns
        -------
        float | complex
            :math:`\\langle\\psi\\vert{O}\\vert\\phi\\rangle` where `O`
            is the matrix-product operator.
        """
        if ket is None:
            ket = bra
        return scprod(bra, self.apply(ket))  # type: ignore

    def reverse(self) -> MPOList:
        """Reverse the sites (see :meth:`~seemps.operators.MPO.reverse`)."""
        return MPOList([o.reverse() for o in self.mpos], self.strategy)


from ..state.simplification import simplify_mps  # noqa: E402
from .mposum import MPOSum  # noqa: E402
