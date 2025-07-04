from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy.sparse as sp  # type: ignore
from typing import TypeAlias, Annotated, TypeVar

Natural: TypeAlias = Annotated[int, ">=1"]

Weight: TypeAlias = float | complex
"""A real or complex number."""

Unitary: TypeAlias = NDArray
"""Unitary matrix in :class:`numpy.ndarray` dense format."""

SparseOperator: TypeAlias = sp.csr_matrix | sp.bsr_matrix | sp.coo_matrix
"""An operator in sparse matrix format."""

Operator: TypeAlias = NDArray | SparseOperator
"""An operator, either in :class:`np.ndarray` or sparse matrix format."""

DenseOperator: TypeAlias = NDArray
"""An operator in :class:`numpy.ndarray` format."""

Vector: TypeAlias = NDArray
"""A one-dimensional :class:`numpy.ndarray` representing a wavefunction."""

VectorLike: TypeAlias = ArrayLike
"""Any Python type that can be coerced to `Vector` type."""

Tensor3: TypeAlias = NDArray
""":class:`numpy.ndarray` tensor with three indices."""

Tensor4: TypeAlias = NDArray
""":class:`numpy.ndarray` tensor with four indices."""

Environment: TypeAlias = NDArray
"""Left or right environment represented as tensor."""

MPOEnvironment: TypeAlias = NDArray
"""Left or right environment of an MPS-MPO-MPS contraction."""

FloatOrArray = TypeVar("FloatOrArray", float, NDArray[np.floating])


def to_dense_operator(O: Operator) -> DenseOperator:
    if sp.issparse(O):
        return O.toarray()  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
    return O


__all__ = [
    "NDArray",
    "Weight",
    "Vector",
    "VectorLike",
    "Operator",
    "Unitary",
    "DenseOperator",
    "Tensor3",
    "Tensor4",
    "Environment",
    "MPOEnvironment",
    "to_dense_operator",
]
