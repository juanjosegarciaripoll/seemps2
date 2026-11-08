from ...typing import Matrix, Vector
from ..mesh import Mesh

def combine_indices(*indices: Matrix, row_major: bool = False) -> Matrix: ...
def maxvol_square(
    A: Matrix, max_iter: int = 10, tol: float = 1.05
) -> tuple[Vector, Matrix]: ...
def maxvol_rectangular(
    A: Matrix,
    rank_kick: tuple[int, int] = (0, 1),
    max_iter: int = 10,
    tol: float = 1.05,
    tol_rect: float = 1.05,
) -> tuple[Vector, Matrix]: ...
def prrlu(
    A: Matrix, tolerance: float, max_rank: int
) -> tuple[Vector, Vector, Matrix, Matrix]: ...
def indices_to_mesh_coordinates(
    mesh: Mesh,
    indices: Matrix | tuple[Matrix, ...],
    map_matrix: Matrix | None = None,
) -> Matrix | None: ...
