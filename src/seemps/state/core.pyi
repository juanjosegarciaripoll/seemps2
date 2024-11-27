from typing import Optional, overload, Iterator, Union, Sequence, Iterable
from ..typing import (
    VectorLike,
    Vector,
    Unitary,
    Tensor3,
    Tensor4,
    Environment,
    Weight,
    Operator,
)
import numpy as np
from numpy.typing import NDArray

MAX_BOND_DIMENSION: int

class Truncation:
    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2
    ABSOLUTE_SINGULAR_VALUE = 3

class Simplification:
    DO_NOT_SIMPLIFY = 0
    CANONICAL_FORM = 1
    VARIATIONAL = 2
    VARIATIONAL_EXACT_GUESS = 3

class Strategy:
    def __init__(
        self: Strategy,
        method: int = Truncation.RELATIVE_SINGULAR_VALUE,
        tolerance: float = 1e-8,
        simplification_tolerance: float = 1e-8,
        max_bond_dimension: int = 0x8FFFFFFF,
        max_sweeps: int = 16,
        normalize: bool = False,
        simplify: int = Simplification.VARIATIONAL,
    ): ...
    def replace(
        self: Strategy,
        method: Optional[int] = None,
        tolerance: Optional[float] = None,
        simplification_tolerance: Optional[float] = None,
        max_bond_dimension: Optional[int] = None,
        max_sweeps: Optional[int] = None,
        normalize: Optional[bool] = None,
        simplify: Optional[int] = None,
    ) -> Strategy: ...
    def set_normalization(self: Strategy, normalize: bool) -> Strategy: ...
    def get_tolerance(self) -> float: ...
    def get_simplification_tolerance(self) -> float: ...
    def get_simplification_method(self) -> Simplification: ...
    def get_max_bond_dimension(self) -> int: ...
    def get_max_sweeps(self) -> int: ...
    def get_normalize_flag(self) -> bool: ...
    def get_simplify_flag(self) -> bool: ...
    def __str__(self) -> str: ...

DEFAULT_TOLERANCE: float

NO_TRUNCATION: Strategy

DEFAULT_STRATEGY: Strategy

def destructively_truncate_vector(s: Vector, strategy: Strategy) -> float: ...
def _contract_nrjl_ijk_klm(U: Unitary, A: Tensor3, B: Tensor3) -> Tensor4: ...
def _contract_last_and_first(A: np.ndarray, B: np.ndarray) -> np.ndarray: ...

class TensorArray(Sequence[NDArray]):
    size: int
    @property
    def _data(self) -> list[Tensor3]: ...
    @_data.setter
    def _data(self, new_data: list[Tensor3]) -> None: ...
    def __init__(self, data: Iterable[NDArray]): ...
    @overload
    def __getitem__(self, k: int) -> NDArray: ...
    @overload
    def __getitem__(self, k: slice) -> Sequence[NDArray]: ...
    def __setitem__(self, k: int, value: NDArray) -> NDArray: ...
    def __iter__(self) -> Iterator[NDArray]: ...
    def __len__(self) -> int: ...

TensorArray3 = TensorArray

class MPS(TensorArray3):
    @classmethod
    def from_vector(
        cls,
        state: VectorLike,
        dimensions: Sequence[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        center: int = -1,
        **kwdargs,
    ) -> MPS: ...
    @classmethod
    def from_tensor(
        cls,
        state: VectorLike,
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        center: int = -1,
        **kwdargs,
    ) -> MPS: ...
    @property
    def __array_priority__(self) -> int: ...
    @property
    def _error(self) -> float: ...
    @_error.setter
    def _error(self, value: float) -> float: ...
    def __init__(self, data: MPS | Sequence[Tensor3], error: float = 0.0): ...
    def copy(self) -> MPS: ...
    def as_mps(self) -> MPS: ...
    def dimension(self) -> int: ...
    def physical_dimensions(self) -> list[int]: ...
    def bond_dimensions(self) -> list[int]: ...
    def max_bond_dimension(self) -> int: ...
    def norm_squared(self) -> float: ...
    def norm(self) -> float: ...
    def zero_state(self) -> MPS: ...
    def left_environment(self, site: int) -> Environment: ...
    def right_environment(self, site: int) -> Environment: ...
    def error(self) -> float: ...
    def set_error(self, error: float) -> float: ...
    def update_error(self, norm2_error_squared: float) -> float: ...
    def conj(self) -> MPS: ...
    @overload
    def __mul__(self, state: MPS) -> MPS: ...
    @overload
    def __mul__(self, weight: Weight) -> MPS: ...
    def __rmul__(self, weight: Weight) -> MPS: ...
    def __add__(self, state: MPS | MPSSum) -> MPSSum: ...
    def __sub__(self, state: MPS | MPSSum) -> MPSSum: ...
    def extend(
        self,
        L: int,
        sites: Optional[Sequence[int]] = None,
        dimensions: Union[int, list[int]] = 2,
        state: Optional[Vector] = None,
    ) -> MPS: ...
    def all_expectation1(self, operator: Union[Operator, list[Operator]]) -> Vector: ...
    def expectation1(self, O: Operator, site: int) -> Weight: ...
    def expectation2(
        self, Opi: Operator, Opj: Operator, i: int, j: Optional[int] = None
    ) -> Weight: ...
    def norm2(self) -> float: ...

class CanonicalMPS(MPS):
    @classmethod
    def from_vector(
        cls,
        state: VectorLike,
        dimensions: Sequence[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        center: int = 0,
        **kwdargs,
    ) -> CanonicalMPS: ...
    def __init__(
        self,
        data: Sequence[Tensor3] | MPS | CanonicalMPS,
        center: int = 0,
        error: float = 0.0,
        normalize: bool = False,
        strategy: Strategy = DEFAULT_STRATEGY,
        is_canonical: bool = False,
    ): ...
    @property
    def center(self) -> int: ...
    def zero_state(self) -> CanonicalMPS: ...
    def norm_squared(self) -> float: ...
    def left_environment(self, site: int) -> Environment: ...
    def right_environment(self, site: int) -> Environment: ...
    def Schmidt_weights(self, site: int) -> Vector: ...
    def entanglement_entropy(self, site: int) -> float: ...
    def Renyi_entropy(self, site: int, alpha: float = 2.0) -> float: ...
    def update_canonical(
        self, A: Tensor3, direction: int, strategy: Strategy
    ) -> float: ...
    def update_2site_right(
        self, AA: Tensor4, site: int, strategy: Strategy
    ) -> float: ...
    def update_2site_left(
        self, AA: Tensor4, site: int, strategy: Strategy
    ) -> float: ...
    def recenter(
        self, site: int, strategy: Strategy = DEFAULT_STRATEGY
    ) -> CanonicalMPS: ...
    def normalize_inplace(self) -> None: ...
    def copy(self) -> CanonicalMPS: ...
    @overload
    def __mul__(self, state: MPS) -> MPS: ...
    @overload
    def __mul__(self, weight: Weight) -> CanonicalMPS: ...
    def __rmul__(self, weight: Weight) -> CanonicalMPS: ...

class MPSSum:
    @property
    def __array_priority__(self) -> int: ...
    @property
    def weights(self) -> list[Weight]: ...
    @property
    def states(self) -> list[MPS]: ...
    @property
    def size(self) -> int: ...
    def __init__(
        self,
        weights: Sequence[Weight],
        states: Sequence[MPS | MPSSum],
        check_args: bool = True,
    ): ...
    def copy(self) -> MPSSum: ...
    def conj(self) -> MPSSum: ...
    def norm_squared(self) -> float: ...
    def norm(self) -> float: ...
    def error(self) -> float: ...
    def physical_dimension(self) -> list[int]: ...
    def dimension(self) -> int: ...
    def __add__(self, state: MPS | MPSSum) -> MPSSum: ...
    def __sub__(self, state: MPS | MPSSum) -> MPSSum: ...
    def __mul__(self, weight: Weight) -> MPSSum: ...
    def __rmul__(self, weight: Weight) -> MPSSum: ...
    def to_vector(self) -> Vector: ...
    def _joined_tensors(self, i: int, L: int) -> Tensor3: ...
    def join(self) -> MPS: ...
    def as_mps(self) -> MPS: ...
    def delete_zero_components(self) -> float: ...

# TODO: hide *environment*() functions with '_'
def _begin_environment(D: Optional[int] = 1) -> Environment: ...
def _update_right_environment(
    B: Tensor3, A: Tensor3, rho: Environment
) -> Environment: ...
def _update_left_environment(
    B: Tensor3, A: Tensor3, rho: Environment
) -> Environment: ...
def _end_environment(rho: Environment) -> Weight: ...
def _join_environments(rhoL: Environment, rhoR: Environment) -> Weight: ...
def scprod(bra: TensorArray3, ket: TensorArray3) -> Weight: ...
def _svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def left_orth_2site(
    AA: Tensor4, strategy: Strategy
) -> tuple[Tensor3, Tensor3, float]: ...
def right_orth_2site(
    AA: Tensor4, strategy: Strategy
) -> tuple[Tensor3, Tensor3, float]: ...
def _select_svd_driver(which: str): ...
def _destructive_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def schmidt_weights(A: np.ndarray) -> np.ndarray: ...
def _update_in_canonical_form_left(
    state: TensorArray3, A: Tensor3, site: int, truncation: Strategy
) -> tuple[int, float]: ...
def _update_in_canonical_form_right(
    state: TensorArray3, A: Tensor3, site: int, truncation: Strategy
) -> tuple[int, float]: ...
def _update_canonical_2site_left(
    state: TensorArray3, A: Tensor4, site: int, truncation: Strategy
) -> float: ...
def _update_canonical_2site_right(
    state: TensorArray3, A: Tensor4, site: int, truncation: Strategy
) -> float: ...
def _canonicalize(state: TensorArray3, center: int, truncation: Strategy) -> float: ...

from .mps import MPS  # noqa: E402
