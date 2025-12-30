import math
import numpy as np
import scipy.sparse as sp
from ..state import Strategy, DEFAULT_STRATEGY
from ..typing import (
    FloatVector,
    SparseOperator,
    DenseOperator,
    Operator,
    to_dense_operator,
)
from ..operators import MPO, CANONICALIZE_MPO, simplify_mpo


class InteractionGraph:
    size: int
    dimensions: list[int]
    _operators: dict[str, DenseOperator]
    _last_key: None | str
    _identity: str
    _interactions: list[str]

    def __init__(self, dimensions: list[int]):
        assert all(isinstance(d, int) and (d > 1) for d in dimensions)
        self.size = len(dimensions)
        self.dimensions = dimensions
        self._last_key = None
        self._operators = dict()
        self._interactions = []
        self._identity = self._add_identities(dimensions)

    def _add_identities(self, dimensions: list[int]) -> str:
        interaction = ""
        for d in dimensions:
            interaction += self.operator_name(np.eye(d))
        return interaction

    @property
    def dimension(self) -> int:
        return math.prod(self.dimensions)

    def add_identical_local_terms(self, O: Operator) -> None:
        for i in range(self.size):
            self.add_local_term(i, O)

    def add_local_term(self, i: int, O: Operator) -> None:
        assert 0 <= i < self.size
        assert O.ndim == 2
        assert O.shape[0] == self.dimensions[i] and O.shape[1] == self.dimensions[i]
        self._interactions.append(
            self._identity[:i] + self.operator_name(O) + self._identity[i + 1 :]
        )

    def add_interaction_term(self, i: int, Oi: Operator, j: int, Oj: Operator) -> None:
        if j < i:
            i, j = j, i
            Oi, Oj = Oj, Oi
        self._interactions.append(
            self._identity[:i]
            + self.operator_name(Oi)
            + self._identity[i + 1 : j]
            + self.operator_name(Oj)
            + self._identity[j + 1 :]
        )

    def add_nearest_neighbor_interaction(
        self, Oi: Operator, Oj: Operator, weights: int | float | FloatVector = 1.0
    ) -> None:
        if isinstance(weights, (float, int)):
            weights = weights * np.ones(self.size - 1)
        assert len(weights) == (self.size - 1)
        for i, w in enumerate(weights):
            self.add_interaction_term(i, Oi, i + 1, w * Oj)

    def add_long_range_interaction(
        self,
        J: Operator,
        Oi: Operator,
        Oj: Operator | None = None,
        keep_diagonals: bool = False,
    ) -> None:
        assert J.shape == (self.size, self.size)
        J = to_dense_operator(J)
        Oi = to_dense_operator(Oi)
        if Oj is None:
            symmetric = True
        else:
            Oj = to_dense_operator(Oj)
            symmetric = bool(np.all(Oi == Oj))
        if symmetric:
            for i in range(J.shape[0]):
                for j in range(i):
                    self.add_interaction_term(i, Oi, j, (J[i, j] + J[j, i]) * Oi)
                if keep_diagonals:
                    self.add_local_term(i, J[i, i] * (Oi @ Oi))
        else:
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    if j != i or keep_diagonals:
                        self.add_interaction_term(i, Oi, j, J[i, j] * Oj)

    def operator_name(self, O: Operator) -> str:
        O = to_dense_operator(O)
        for key, value in self._operators.items():
            if (O.shape == value.shape) and np.allclose(O, value):
                return key
        key = self.new_key()
        self._operators[key] = O
        return key

    def new_key(self) -> str:
        if self._last_key is None:
            self._last_key = "a"
        else:
            self._last_key = chr(ord(self._last_key) + 1)
        return self._last_key

    def to_mpo(
        self, strategy: Strategy = DEFAULT_STRATEGY, simplify: bool = True
    ) -> MPO:
        def all_prefixes(site: int) -> dict[str, int]:
            output = dict()
            n = 0
            for term in self._interactions:
                prefix = term[:site]
                if prefix not in output:
                    output[prefix] = n
                    n += 1
            return output

        def all_suffixes(site: int) -> dict[str, int]:
            output = dict()
            n = 0
            for term in self._interactions:
                suffix = term[site:]
                if suffix not in output:
                    output[suffix] = n
                    n += 1
            return output

        tensors = []
        for i in range(self.size):
            L = all_prefixes(i)
            R = all_prefixes(i + 1)
            A = np.zeros(
                (len(L), self.dimensions[i], self.dimensions[i], len(R)),
                dtype=np.complex128,
            )
            for term in self._interactions:
                iL = L[term[:i]]
                iR = R[term[: i + 1]]
                O = self._operators[term[i]]
                A[iL, :, :, iR] = O
            tensors.append(A.real if np.all(A.imag == 0.0) else A)
        lastA = tensors[-1]
        tensors[-1] = np.sum(lastA, -1).reshape(lastA.shape[:-1] + (1,))
        mpo = MPO(tensors, strategy)
        if simplify:
            return simplify_mpo(mpo, CANONICALIZE_MPO, direction=+1)
        return mpo

    def to_matrix(self) -> SparseOperator:
        def build_sparse_matrix(term: str) -> SparseOperator:
            output = sp.identity(1).tobsr()  # type: ignore
            for name in term:
                output = sp.kron(output, sp.bsr_matrix(self._operators[name]))
            return output.tocsr()

        return sum(
            (build_sparse_matrix(term) for term in self._interactions),
            start=sp.csr_matrix((self.dimension, self.dimension)),
        )
