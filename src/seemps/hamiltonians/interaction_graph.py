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
from ..operators import MPO
from ..cython import _canonicalize


class InteractionGraph:
    r"""
    Proxy object to help in the construction of MPOs from local terms and
    arbitrary interactions.

    This class allows the user to keep track of all interactions in a complex
    Hamiltonian, adding local terms

    .. math::
        \sum_i h_i O_i

    nearest-neighbor interactions

    .. math::
        \sum_i J_i O_i O_{i+1}

    or arbitrary long-range terms

    .. math::
        \sum_{ij} J_{ij} O_i O_j

    Parameters
    ----------
    dimensions : list[int]
        List of dimensions of the quantum objects involved

    Attributes
    ----------
    size : int
        Number of quantum objects
    dimensions : list[int]
        List of dimensions
    dimension: int
        Total dimension of the Hilbert space (if it can be computed).
    """

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
            interaction += self._operator_name(np.eye(d))
        return interaction

    def _operator_name(self, O: Operator) -> str:
        O = to_dense_operator(O)
        for key, value in self._operators.items():
            if (O.shape == value.shape) and np.allclose(O, value):
                return key
        key = self._new_key()
        self._operators[key] = O
        return key

    def _new_key(self) -> str:
        if self._last_key is None:
            self._last_key = "a"
        else:
            self._last_key = chr(ord(self._last_key) + 1)
        return self._last_key

    @property
    def dimension(self) -> int:
        return math.prod(self.dimensions)

    def add_identical_local_terms(self, O: Operator) -> None:
        r"""Add a sum of local terms :math:`\sum_i O`."""
        for i in range(self.size):
            self.add_local_term(i, O)

    def add_local_term(self, i: int, O: Operator) -> None:
        """Add a single local term `O` acting on the i-th component."""
        assert 0 <= i < self.size
        assert O.ndim == 2
        assert O.shape[0] == self.dimensions[i] and O.shape[1] == self.dimensions[i]
        self._interactions.append(
            self._identity[:i] + self._operator_name(O) + self._identity[i + 1 :]
        )

    def add_interaction_term(self, i: int, Oi: Operator, j: int, Oj: Operator) -> None:
        """Add a pair-wise interaction between sites `i` and `j` with respective operators `Oi` and `Oj`."""
        if j < i:
            i, j = j, i
            Oi, Oj = Oj, Oi
        self._interactions.append(
            self._identity[:i]
            + self._operator_name(Oi)
            + self._identity[i + 1 : j]
            + self._operator_name(Oj)
            + self._identity[j + 1 :]
        )

    def add_nearest_neighbor_interaction(
        self, A: Operator, B: Operator, weights: int | float | FloatVector = 1.0
    ) -> None:
        r"""Add a nearest-neighbor interaction sum :math:`\sum_{i} w_{i} A_i B_{i+1}`.

        Parameters
        ----------
        A, B : Operator
            These are the operators :math:`A` and :math:`B`.
        weights : int | float | FloatVector
            The list of weights, or a common weight :math:`w_i=w` for all.
            Defaults to :math:`w_i=1`.
        """
        if isinstance(weights, (float, int)):
            weights = weights * np.ones(self.size - 1)
        assert len(weights) == (self.size - 1)
        for i, w in enumerate(weights):
            self.add_interaction_term(i, A, i + 1, w * B)

    def add_long_range_interaction(
        self,
        J: Operator,
        A: Operator,
        B: Operator | None = None,
        keep_diagonals: bool = False,
    ) -> None:
        r"""Add a nearest-neighbor interaction sum :math:`\sum_{i} J_{ij} A_i B_{i+1}`.

        Parameters
        ----------
        J : Operator
            The list of weights, or a common weight :math:`w_i=w` for all.
            Defaults to :math:`w_i=1`.
        A, B : Operator
            These are the operators :math:`A` and :math:`B`.
        keep_diagonals : bool
            If False, the terms :math:`A_iB_i` are not included (defaults to False).
        """
        assert J.shape == (self.size, self.size)
        J = to_dense_operator(J)
        A = to_dense_operator(A)
        if B is None:
            symmetric = True
        else:
            B = to_dense_operator(B)
            symmetric = bool(np.all(A == B))
        if symmetric:
            for i in range(J.shape[0]):
                for j in range(i):
                    self.add_interaction_term(i, A, j, (J[i, j] + J[j, i]) * A)
                if keep_diagonals:
                    self.add_local_term(i, J[i, i] * (A @ A))
        else:
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    if j != i or keep_diagonals:
                        self.add_interaction_term(i, A, j, J[i, j] * B)

    def to_mpo(
        self,
        strategy: Strategy = DEFAULT_STRATEGY,
        simplify: bool = True,
        simplification_strategy: Strategy = DEFAULT_STRATEGY,
    ) -> MPO:
        """Construct the MPO associated to these interactions."""

        def all_prefixes(site: int) -> dict[str, int]:
            output = dict()
            n = 0
            for term in self._interactions:
                prefix = term[:site]
                if prefix not in output:
                    output[prefix] = n
                    n += 1
            return output

        def all_local_terms(site: int) -> set[str]:
            return set(term[i] for term in self._interactions)

        state = []
        local_operators = []
        L = all_prefixes(0)
        for i in range(self.size):
            C = {name: index for index, name in enumerate(all_local_terms(i))}
            R = all_prefixes(i + 1)
            A = np.zeros((len(L), len(C), len(R)))
            for term in self._interactions:
                iL = L[term[:i]]
                iC = C[term[i]]
                iR = R[term[: i + 1]]
                A[iL, iC, iR] = 1
            local_operators.append(C)
            state.append(A)
            L = R
        lastA = state[-1]
        state[-1] = np.sum(lastA, -1).reshape(lastA.shape[:-1] + (1,))
        if simplify:
            _canonicalize(state, 0, simplification_strategy)

        tensors = []
        for d, C, A in zip(self.dimensions, local_operators, state):
            a, _, b = A.shape
            B = np.zeros((A.shape[0], d, d, A.shape[-1]), dtype=np.complex128)
            for name, index in C.items():
                B += A[:, index, :].reshape(a, 1, 1, b) * self._operators[name].reshape(
                    d, d, 1
                )
            tensors.append(B.real if np.all(B.imag == 0.0) else B)
        return MPO(tensors, strategy)

    def to_matrix(self) -> SparseOperator:
        """Construct the sparse matrix associated to these interactions."""

        def build_sparse_matrix(term: str) -> SparseOperator:
            output = sp.identity(1).tobsr()  # type: ignore
            for name in term:
                output = sp.kron(output, sp.bsr_matrix(self._operators[name]))
            return output.tocsr()

        return sum(
            (build_sparse_matrix(term) for term in self._interactions),
            start=sp.csr_matrix((self.dimension, self.dimension)),
        )
