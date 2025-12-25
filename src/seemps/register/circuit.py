from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from math import sqrt
from ..typing import DenseOperator, Operator, Real
from ..state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from ..cython import _contract_nrjl_ijk_klm
from abc import abstractmethod, ABC

σx = np.array([[0.0, 1.0], [1.0, 0.0]])
σz = np.array([[1.0, 0.0], [0.0, -1.0]])
σy = -1j * σz @ σx
id2 = np.eye(2)
CNOT = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

Parameters = Sequence[Real] | np.ndarray[tuple[int], np.dtype[np.floating]]

known_operators: dict[str, DenseOperator] = {
    "SX": σx / 2.0,
    "SY": σy / 2.0,
    "SZ": σz / 2.0,
    "σX": σx,
    "σY": σy,
    "σZ": σz,
    "CNOT": CNOT,
    "CX": CNOT,
    "CZ": np.diag([1.0, 1.0, 1.0, -1.0]),
}


def interpret_operator(op: str | Operator) -> DenseOperator:
    O: Operator
    if isinstance(op, str):
        O = known_operators[op.upper()]
    elif not isinstance(op, np.ndarray) or op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise Exception(f"Invalid qubit operator of type '{type(op)}")
    else:
        O = op
    return O


class UnitaryCircuit(ABC):
    register_size: int
    strategy: Strategy

    def __init__(
        self,
        register_size: int,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.strategy = strategy
        self.register_size = register_size

    @abstractmethod
    def apply_inplace(
        self, state: MPS, parameters: Parameters | None = None
    ) -> CanonicalMPS: ...

    def __matmul__(self, state: MPS) -> CanonicalMPS:
        return self.apply(state)

    def apply(self, state: MPS, parameters: Parameters | None = None) -> CanonicalMPS:
        return self.apply_inplace(
            state.copy() if isinstance(state, CanonicalMPS) else state, parameters
        )


class ParameterizedCircuit(UnitaryCircuit, ABC):
    parameters: np.ndarray[tuple[int], np.dtype[np.floating]]
    parameters_size: int

    def __init__(
        self,
        register_size: int,
        parameters_size: int | None = None,
        default_parameters: Parameters | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        super().__init__(register_size, strategy)
        if default_parameters is None:
            if parameters_size is None:
                raise Exception(
                    "In ParameterizedUnitaries, either parameter_size or default_parameters must be provided"
                )
            self.parameters = np.zeros(parameters_size)
        else:
            self.parameters = np.asarray(default_parameters)
            if parameters_size is None:
                parameters_size = len(default_parameters)
            elif parameters_size != len(default_parameters):
                raise IndexError(
                    f"'default_parameters' length {len(default_parameters)} does not match size 'parameters_size' {parameters_size}"
                )
        self.parameters_size = parameters_size


class LocalRotationsLayer(ParameterizedCircuit):
    """Layer of local rotations acting on the each qubit with the
    same generator and possibly different angles.

    Parameters
    ----------
    register_size : int
        Number of qubits on which to operate.
    operator : str | Operator
        Either the name of a generator ("Sx", "Sy", "Sz") or a 2x2 matrix.
    same_parameter : bool
        If `True`, the same angle is reused by all gates and
        `self.parameters_size=1`. Otherwise, the user must provide one value
        for each rotation.
    default_parameters : Sequence[Real] | None
        A list or a vector of angles to use if no other one is provided.
    strategy : Strategy
        Truncation and simplification strategy (Defaults to `DEFAULT_STRATEGY`)

    Examples
    --------
    >>> state = random_uniform_mps(2, 3)
    >>> U = LocalRotationsLayer(register_size=state.size, operator="Sz")
    >>> Ustate = U @ state
    """

    factor: float = 1.0
    operator: DenseOperator

    def __init__(
        self,
        register_size: int,
        operator: str | DenseOperator,
        same_parameter: bool = False,
        default_parameters: Parameters | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        if same_parameter:
            parameters_size = 1
            if default_parameters is not None:
                if len(default_parameters) > 1:
                    raise Exception(
                        "Cannot provide more than one parameter if same_parameter is True"
                    )
        else:
            parameters_size = register_size
        super().__init__(
            register_size,
            parameters_size,
            default_parameters,
            strategy,
        )
        O = interpret_operator(operator)
        if O.shape != (2, 2):
            raise Exception("Not a valid one-qubit operator")
        #
        # self.operator is a Pauli operator with det = 1. We
        # extract the original determinant into a prefactor for the
        # rotation angles.
        #
        self.factor = sqrt(abs(np.linalg.det(O)))
        self.operator = O / self.factor

    def apply_inplace(
        self, state: MPS, parameters: Parameters | None = None
    ) -> CanonicalMPS:
        assert self.register_size == state.size
        if parameters is None:
            parameters = self.parameters
        if len(parameters) == 1:
            angle = np.full(self.register_size, parameters[0])
        else:
            angle = np.asarray(parameters)
        if not isinstance(state, CanonicalMPS):
            state = CanonicalMPS(state, center=0, strategy=self.strategy)
        angle = angle.reshape(-1, 1, 1) * self.factor
        ops = np.cos(angle) * id2 - 1j * np.sin(angle) * self.operator
        for i, (opi, A) in enumerate(zip(ops, state)):
            # np.einsum('ij,ajb->aib', opi, A)
            state[i] = np.matmul(opi, A)
        return state


class TwoQubitGatesLayer(UnitaryCircuit):
    """Layer of CNOT/CZ gates, acting on qubits 0 to N-1, from left to right
    or right to left, depending on the direction.

    Parameters
    ----------
    register_size : int
        Number of qubits on which to operate.
    operator : str | Operator
        A two-qubit gate, either denoted by a string ("CNOT", "CZ")
        or by a 4x4 two-qubit matrix.
    direction : int | None
        Direction in which gates are applied. If 'None', direction will
        be chosen based on the orthogonalitiy center of the state.
    strategy : Strategy
        Truncation strategy (Defaults to `DEFAULT_STRATEGY`)
    """

    operator: DenseOperator
    direction: int | None

    def __init__(
        self,
        register_size: int,
        operator: str | Operator,
        direction: int | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        super().__init__(register_size, strategy)
        O = interpret_operator(operator)
        if O.shape != (4, 4):
            raise Exception("Not a valid two-qubit operator")
        self.operator = O
        self.direction = direction

    def apply_inplace(
        self, state: MPS, parameters: Parameters | None = None
    ) -> CanonicalMPS:
        assert self.register_size == state.size
        if parameters is not None and len(parameters) > 0:
            raise Exception("{self.cls} does not accept parameters")
        if not isinstance(state, CanonicalMPS):
            state = CanonicalMPS(state, center=0, strategy=self.strategy)
        L = self.register_size
        op = self.operator
        center = state.center
        strategy = self.strategy
        direction = self.direction
        if direction is None:
            direction = +1 if (center < L // 2) else -1
        if direction >= 0:
            if center > 1:
                state.recenter(1)
            for j in range(L - 1):
                state.update_2site_right(
                    _contract_nrjl_ijk_klm(op, state[j], state[j + 1]), j, strategy
                )
        else:
            if center < L - 2:
                state.recenter(L - 2)
            for j in range(L - 2, -1, -1):
                # AA = np.einsum("ijk,klm,nrjl -> inrm", state[j], state[j + 1], U[j])
                state.update_2site_left(
                    _contract_nrjl_ijk_klm(op, state[j], state[j + 1]), j, strategy
                )
        return state


class ParameterizedLayeredCircuit(ParameterizedCircuit):
    """Variational quantum circuit with Ry rotations and CNOTs.

    Constructs a unitary circuit with variable parameters, composed of
    operations such as :class:`LocalRotationsLayer`, :class:`TwoQubitGatesLayer`
    or similar gates. This is the basis for more useful algorithms such as
    the :class:`VariationalQuantumEigensolver`

    Parameters
    ----------
    register_size : int
        Number of qubits on which to operate
    layers : list[UnitaryCircuit]
        List of constant or parameterized unitary layers.
    default_parameters : Sequence[Real]
        Default angles for the rotations (Defaults to zeros).
    strategy : Strategy
        Truncation and simplification strategy (Defaults to `DEFAULT_STRATEGY`)
    """

    layers: list[tuple[UnitaryCircuit, int, int]]

    def __init__(
        self,
        register_size: int,
        layers: list[UnitaryCircuit],
        default_parameters: Parameters | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        parameters_size = 0
        segments: list[tuple[UnitaryCircuit, int, int]] = []
        for circuit in layers:
            if isinstance(circuit, ParameterizedCircuit):
                l = circuit.parameters_size
                segments.append((circuit, parameters_size, parameters_size + l))
                parameters_size += l
            else:
                segments.append((circuit, 0, 0))
        super().__init__(register_size, parameters_size, default_parameters, strategy)
        self.layers = segments

    def apply_inplace(
        self, state: MPS, parameters: Parameters | None = None
    ) -> CanonicalMPS:
        if parameters is None:
            parameters = self.parameters
        for circuit, start, end in self.layers:
            state = circuit.apply_inplace(state, parameters[start:end])
        if not isinstance(state, CanonicalMPS):
            return CanonicalMPS(state)
        return state  # type: ignore


class VQECircuit(ParameterizedLayeredCircuit):
    """Variational quantum circuit with Ry rotations and CNOTs.

    Parameters
    ----------
    register_size : int
        Number of qubits on which to operate
    layers : int
        Number of local rotation layers
    default_parameters : Vector
        Default angles for the rotations (Defaults to zeros). Must have
        size `layers * register_size`.
    strategy : Strategy
        Truncation and simplification strategy (Defaults to `DEFAULT_STRATEGY`)
    """

    def __init__(
        self,
        register_size: int,
        layers: int,
        default_parameters: Parameters | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        parameters_seq: list[None] | np.ndarray
        if default_parameters is None:
            parameters_seq = [None] * layers
        else:
            parameters_seq = np.asarray(default_parameters).reshape(-1, register_size)

        super().__init__(
            register_size,
            [
                LocalRotationsLayer(
                    register_size,
                    operator="Sy",
                    same_parameter=False,
                    default_parameters=parameters_seq[layer // 2],
                    strategy=strategy,
                )
                if (layer % 2 == 0)
                else TwoQubitGatesLayer(
                    register_size,
                    operator="CNOT",
                    direction=+1 if (layer % 4) == 1 else -1,
                )
                for layer in range(2 * layers)
            ],
            default_parameters,
            strategy,
        )
