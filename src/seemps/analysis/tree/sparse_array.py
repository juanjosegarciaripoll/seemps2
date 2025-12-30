from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from typing import TypeAlias
from ...typing import Tensor3
from collections.abc import Sequence, Iterable, Iterator
from typing import overload, TypeVar

_T = TypeVar("_T", bound="SparseTensorArray")


class SparseCore:
    """
    Sparse rank-3 MPS tensor core.

    Stores a physical slice for each physical index as a CSR matrix, reducing memory and
    accelerating contractions when the structure is sparse. All slices share the same (r_L, r_R) shape.
    """

    data: list[sp.csr_array]
    shape: tuple[int, int, int]

    def __init__(self, data: list[sp.csr_array]):
        r_L, r_R = data[0].shape
        s = len(data)
        for matrix in data:
            if matrix.shape != (r_L, r_R):
                raise ValueError("All tensor slices must be of the same shape.")

        self.data = data
        self.shape = (r_L, s, r_R)

    def conj(self) -> SparseCore:
        return SparseCore([matrix.conj() for matrix in self.data])

    def transpose(self) -> SparseCore:
        return SparseCore([A.T.tocsr() for A in self.data])

    def to_dense(self) -> Tensor3:
        core = np.zeros(self.shape, dtype=np.float64)
        for idx, matrix in enumerate(self.data):
            core[:, idx, :] = matrix.toarray()
        return core


SparseMPSTensor: TypeAlias = SparseCore | Tensor3


class SparseTensorArray(Sequence[SparseMPSTensor]):
    """TensorArray class.

    This class provides the basis for all tensor networks. The class abstracts
    a one-dimensional array of tensors that is freshly copied whenever the
    object is cloned. Two TensorArray's can share the same tensors and be
    destructively modified.

    Parameters
    ----------
    data: Iterable[SparseMPSTensor]
        Any sequence of tensors that can be stored in this object. They are
        not checked to have the right number of dimensions. This sequence is
        cloned to avoid nasty side effects when destructively modifying it.
    """

    _data: list[SparseMPSTensor]
    size: int

    def __init__(self, data: Iterable[SparseMPSTensor]):
        self._data = list(data)
        self.size = len(self._data)

    @overload
    def __getitem__(self, k: int) -> SparseMPSTensor: ...

    @overload
    def __getitem__(self, k: slice) -> Sequence[SparseMPSTensor]: ...

    def __getitem__(
        self, k: int | slice
    ) -> SparseMPSTensor | Sequence[SparseMPSTensor]:
        #
        # Get MP matrix at position `k`. If 'A' is an MP, we can now
        # do A[k]
        #
        return self._data[k]  # type: ignore

    @overload
    def __setitem__(self, k: int, value: SparseMPSTensor) -> SparseMPSTensor: ...

    @overload
    def __setitem__(
        self, k: slice, value: Sequence[SparseMPSTensor]
    ) -> Sequence[SparseMPSTensor]: ...

    def __setitem__(
        self, k: int | slice, value: SparseMPSTensor | Sequence[SparseMPSTensor]
    ) -> SparseMPSTensor | Sequence[SparseMPSTensor]:
        #
        # Replace matrix at position `k` with new tensor `value`. If 'A'
        # is an MP, we can now do A[k] = value
        #
        if isinstance(k, slice):
            self._data[k] = list(value)  # type: ignore
        else:
            self._data[k] = value  # type: ignore # pyright: ignore[reportCallIssue, reportArgumentType]
        return value

    def __iter__(self) -> Iterator[SparseMPSTensor]:
        return self._data.__iter__()

    def __reversed__(self) -> Iterator[SparseMPSTensor]:
        return self._data.__reversed__()

    def __len__(self) -> int:
        return self.size
