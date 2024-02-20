from __future__ import annotations
from numpy.typing import NDArray, ArrayLike
import scipy.sparse  # type: ignore
from typing import TypeAlias, Union

Weight: TypeAlias = Union[float, complex]
"""A real or complex number."""

Unitary: TypeAlias = NDArray
"""Unitary matrix in :class:`numpy.ndarray` dense format."""

Operator: TypeAlias = Union[NDArray, scipy.sparse.sparray]
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
]
