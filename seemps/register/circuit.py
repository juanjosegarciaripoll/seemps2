import numpy as np
from ..typing import *
from ..state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from ..state._contractions import _contract_nrjl_ijk_klm
from abc import abstractmethod

σx = np.array([[0.0, 1.0], [1.0, 0.0]])
σz = np.array([[1.0, 0.0], [0.0, -1.0]])
σy = -1j * σz @ σx
id2 = np.eye(2)

known_operators = {
    "Sx": σx / 2.0,
    "Sy": σy / 2.0,
    "Sz": σz / 2.0,
    "σx": σx,
    "σy": σy,
    "σz": σz,
    "CNOT": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    "CZ": np.diag([1.0, 1.0, 1.0, -1.0]),
}


def interpret_operator(op: Union[str, Operator]) -> Operator:
    O: Operator
    if isinstance(op, str):
        O = known_operators.get(op, None)
        if O is None:
            raise Exception(f"Unknown qubit operator '{op}'")
    elif not isinstance(op, np.ndarray):
        raise Exception(f"Invalid qubit operator of type '{type(op)}")
    else:
        O = op
    return O


class UnitaryCircuit:
    register_size: int
    stategy: Strategy

    def __init__(
        self,
        register_size: int,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.strategy = strategy
        self.register_size = register_size

    @abstractmethod
    def apply_inplace(
        self, state: MPS, parameters: Optional[Vector] = None
    ) -> CanonicalMPS:
        ...

    def apply(self, state: MPS, parameters: Optional[Vector] = None) -> CanonicalMPS:
        return self.apply_inplace(
            state.copy() if isinstance(state, CanonicalMPS) else state, parameters
        )


class ParameterizedCircuit(UnitaryCircuit):
    register_size: int
    parameters: Vector
    stategy: Strategy

    def __init__(
        self,
        register_size: int,
        parameters_size: Optional[int] = None,
        default_parameters: Optional[Vector] = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        super().__init__(register_size, strategy)
        if default_parameters is None:
            if parameters_size is None:
                raise Exception(
                    "In ParameterizedUnitaries, either parameter_size or default_parameters must be provided"
                )
            default_parameters = np.zeros(parameters_size)
        elif parameters_size != len(default_parameters):
            raise IndexError(
                "'default_parameters' does not match size 'parameter_size'"
            )
        self.parameters = np.asarray(default_parameters)
        self.parameters_size = parameters_size


class LocalRotationsLayer(ParameterizedCircuit):
    factor: float = 1.0
    operator: Operator

    def __init__(
        self,
        register_size: int,
        operator: Union[str, Operator],
        same_parameter: bool = False,
        default_parameters: Optional[Vector] = None,
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
        factor = np.abs(np.linalg.det(O))
        self.factor = np.sqrt(abs(factor))
        self.operator = O / self.factor

    def apply_inplace(
        self, state: MPS, parameters: Optional[Vector] = None
    ) -> CanonicalMPS:
        assert self.register_size == state.size
        if parameters is None:
            parameters = self.parameters
        if not isinstance(state, CanonicalMPS):
            state = CanonicalMPS(state, center=0, strategy=self.strategy)
        if len(parameters) == 1:
            parameters = np.full(self.register_size, parameters[0])
        angle = parameters.reshape(-1, 1, 1) * self.factor
        ops = np.cos(angle) * id2 - 1j * np.sin(angle) * self.operator
        for i, (opi, A) in enumerate(zip(ops, state)):
            # np.einsum('ij,ajb->aib', opi, A)
            state[i] = np.matmul(opi, A)
        return state


class TwoQubitGatesLayer(UnitaryCircuit):
    operator: Operator

    def __init__(
        self,
        register_size: int,
        operator: Union[str, Operator],
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        super().__init__(register_size, strategy)
        O = interpret_operator(operator)
        if O.shape != (4, 4):
            raise Exception("Not a valid two-qubit operator")
        self.operator = O

    def apply_inplace(
        self, state: MPS, parameters: Optional[Vector] = None
    ) -> CanonicalMPS:
        assert self.register_size == state.size
        if parameters is not None:
            if len(parameters) > 0:
                raise Exception("{self.cls} does not accept parameters")
        if not isinstance(state, CanonicalMPS):
            state = CanonicalMPS(state, center=0, strategy=self.strategy)
        L = self.register_size
        op = self.operator
        center = state.center
        strategy = self.strategy
        if center < L // 2:
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
                ## AA = np.einsum("ijk,klm,nrjl -> inrm", state[j], state[j + 1], U[j])
                state.update_2site_left(
                    _contract_nrjl_ijk_klm(op, state[j], state[j + 1]), j, strategy
                )
        return state
