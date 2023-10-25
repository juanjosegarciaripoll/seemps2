from __future__ import annotations

import copy

import numpy as np
import opt_einsum  # type: ignore

from ..state import (DEFAULT_STRATEGY, MPS, CanonicalMPS, MPSSum, Strategy,
                     Weight, array)
from ..state.environments import *
from ..tools import InvalidOperation, log
from ..truncate import combine
from ..typing import *


def _mpo_multiply_tensor(A, B):
    # Implements
    # np.einsum("cjd,aijb->caidb", A, B)
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


class MPO(array.TensorArray):
    """Matrix Product Operator class.

    This implements a bare-bones Matrix Product Operator object with open
    boundary conditions. The tensors have four indices, A[α,i,j,β], where
    'α,β' are the internal labels and 'i,j' the physical indices ar the given
    site.

    Parameters
    ----------
    data: list[Tensor4]
        List of four-legged tensors forming the structure.
    strategy: Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for algorithms.
    """

    strategy: Strategy

    __array_priority__ = 10000

    def __init__(self, data: list[Tensor4], strategy: Strategy = DEFAULT_STRATEGY):
        super(MPO, self).__init__(data)
        assert data[0].shape[0] == data[-1].shape[-1] == 1
        self.strategy = strategy

    def __add__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self + A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, 1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + A.weights, A.strategy)
        raise TypeError(f"Cannod add MPO and {type(A)}")

    def __sub__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self - A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, -1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + [-w for w in A.weights], A.strategy)
        raise TypeError(f"Cannod subtract MPO and {type(A)}")

    # TODO: The deep copy also copies the tensors. This should be improved.
    def __mul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `n * self`"""
        if isinstance(n, (int, float, complex)):
            mpo_mult = copy.deepcopy(self)
            mpo_mult._data[0] = n * mpo_mult._data[0]
            return mpo_mult
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `self * self`"""
        if isinstance(n, (int, float, complex)):
            mpo_mult = copy.deepcopy(self)
            mpo_mult._data[0] = n * mpo_mult._data[0]
            return mpo_mult
        raise InvalidOperation("*", n, self)

    # TODO: Rename to physical_dimensions()
    def dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPO."""
        return [A.shape[1] for A in self._data]

    # TODO: Rename to to_matrix()
    def tomatrix(self) -> Operator:
        """Convert this MPO to a dense or sparse matrix."""
        D = 1  # Total physical dimension so far
        out = np.array([[[1.0]]])
        for A in self._data:
            _, i, _, b = A.shape
            out = np.einsum("lma,aijb->limjb", out, A)
            D *= i
            out = out.reshape(D, D, b)
        return out[:, :, 0]

    def apply(
        self,
        b: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> MPS:
        """Implement multiplication `A @ b` between a matrix-product operator
        `A` and a matrix-product state `b`.

        Parameters
        ----------
        b : MPS | MPSSum
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
        if isinstance(b, MPSSum):
            state: MPS = combine(weights=b.weights, states=b.states, truncation=strategy)
        elif isinstance(b, MPS):
            state = b
        else:
            raise TypeError(f"Cannot multiply MPO with {b}")
        assert self.size == state.size
        state = MPS(
            [_mpo_multiply_tensor(A, B) for A, B in zip(self._data, state._data)],
            error=state.error(),
        )
        if simplify:
            state = truncate.simplify(
                state, truncation=strategy
            )
        return state

    def __matmul__(self, b: Union[MPS, MPSSum]) -> MPS:
        """Implement multiplication `self @ b`."""
        if isinstance(b, (MPS, MPSSum)):
            return self.apply(b)
        raise InvalidOperation("@", self, b)

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Optional[Sequence[int]] = None,
        dimensions: Union[int, list[int]] = 2,
    ) -> MPO:
        """Enlarge an MPO so that it acts on a larger Hilbert space with 'L' sites.

        Parameters
        ----------
        L : int
            The new size of the MPS. Must be strictly larger than `self.size`.
        sites : Iterable[int], optional
            Sequence of integers describing the sites that occupied by the
            tensors in this state.
        dimensions : Union[int, list[int]], default = 2
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

    def expectation(self, bra: MPS, ket: Optional[MPS] = None) -> Weight:
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
        ket : Optional[MPS]
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
        operators = self._data
        for i in range(0, center):
            left = update_left_mpo_environment(
                left, bra[i].conj(), operators[i], ket[i]
            )
        for i in range(self.size - 1, center - 1, -1):
            right = update_right_mpo_environment(
                right, bra[i].conj(), operators[i], ket[i]
            )
        return join_mpo_environments(left, right)


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

    __array_priority__ = 10000

    mpos: list[MPO]
    strategy: Strategy
    size: int

    def __init__(self, mpos: Sequence[MPO], strategy: Strategy = DEFAULT_STRATEGY):
        assert len(mpos) > 1
        self.mpos = mpos = list(mpos)
        self.size = mpos[0].size
        self.strategy = strategy

    def __add__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self + A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, 1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + A.weights, A.strategy)
        raise TypeError(f"Cannod add MPO and {type(A)}")

    def __sub__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
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

    # TODO: Rename to to_matrix()
    def tomatrix(self) -> Operator:
        """Convert this MPO to a dense or sparse matrix."""
        A = self.mpos[0].tomatrix()
        for mpo in self.mpos[1:]:
            A = mpo.tomatrix() @ A
        return A

    # TODO: Describe how `strategy` and simplify act as compared to
    # the values provided by individual operators.
    def apply(
        self,
        b: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
        simplify: Optional[bool] = None,
    ) -> MPS:
        """Implement multiplication `A @ b` between a matrix-product operator
        `A` and a matrix-product state `b`.

        Parameters
        ----------
        b : MPS | MPSSum
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
        state: MPS
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        if isinstance(b, MPSSum):
            state: MPS = combine(weights=b.weights, states=b.states, truncation=strategy)
        else:
            state = b
        for mpo in self.mpos:
            # log(f'Total error before applying MPOList {b.error()}')
            state = mpo.apply(state)
        if simplify:
            state = truncate.simplify(
                state, truncation=strategy
            )
        return state

    def __matmul__(self, b: Union[MPS, MPSSum]) -> MPS:
        """Implement multiplication `self @ b`."""
        if isinstance(b, (MPS, MPSSum)):
            return self.apply(b)
        raise InvalidOperation("@", self, b)

    def extend(
        self, L: int, sites: Optional[list[int]] = None, dimensions: int = 2
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

        def join(A, *args):
            if not args:
                return A
            B = join(*args)
            a, d, d, b = A.shape
            c, d, d, e = B.shape
            # A, B, args[1],... are the tensors of the MPO to
            # join. They are applied to the MPS in this order, hence the
            # particular position of elements in opt_einsum
            # TODO: Remove dependency on opt_einsum
            return opt_einsum.contract("aijb,cjkd->acikbd", B, A).reshape(
                a * c, d, d, b * e
            )

        return join(*[mpo[i] for mpo in self.mpos])

    def join(self, strategy: Optional[Strategy] = None) -> MPO:
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


from .. import truncate
from .mposum import MPOSum
