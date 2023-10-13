from ..typing import Optional, Vector

MAX_BOND_DIMENSION: int

class Truncation:
    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2
    ABSOLUTE_SINGULAR_VALUE = 3

class Simplification:
    CANONICAL_FORM = 0
    VARIATIONAL = 1

class Strategy:
    def __init__(
        self: Strategy,
        method: int = 1,
        simplification_method: int  = 1,
        tolerance: float = 1e-8,
        max_bond_dimension: int = 0x8FFFFFFF,
        max_sweeps: int = 16,
        normalize: bool = False,
        simplify: bool = False,
    ): ...
    def replace(
        self: Strategy,
        method: Optional[int] = None,
        simplification_method: Optional[int] = None,
        tolerance: Optional[float] = None,
        max_bond_dimension: Optional[int] = None,
        max_sweeps: Optional[int] = None,
        normalize: Optional[bool] = None,
        simplify: Optional[bool] = None,
    ) -> Strategy: ...
    def set_normalization(self: Strategy, normalize: bool) -> Strategy: ...
    def get_tolerance(self) -> float: ...
    def get_max_bond_dimension(self) -> int: ...
    def get_max_sweeps(self) -> int: ...
    def get_normalize_flag(self) -> bool: ...
    def get_simplify_flag(self) -> bool: ...
    def get_simplification_method(self) -> int: ...
    def __str__(self) -> str: ...

DEFAULT_TOLERANCE: float

NO_TRUNCATION: Strategy

DEFAULT_STRATEGY: Strategy

def truncate_vector(s: Vector, strategy: Strategy) -> tuple[Vector, float]:
    pass
