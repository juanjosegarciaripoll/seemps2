from typing import Optional
from ..typing import Vector, Unitary, Tensor3, Tensor4
import numpy as np

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
    VARIATIONAL_EXACT = 3

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

def truncate_vector(s: Vector, strategy: Strategy) -> tuple[Vector, float]: ...
def _contract_nrjl_ijk_klm(U: Unitary, A: Tensor3, B: Tensor3) -> Tensor4: ...
def _contract_last_and_first(A: np.ndarray, B: np.ndarray) -> np.ndarray: ...
