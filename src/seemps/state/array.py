from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import overload, TypeVar, Sequence, Iterable, Union, Iterator

_T = TypeVar("_T", bound="TensorArray")


class TensorArray(Sequence[NDArray]):
    """TensorArray class.

    This class provides the basis for all tensor networks. The class abstracts
    a one-dimensional array of tensors that is freshly copied whenever the
    object is cloned. Two TensorArray's can share the same tensors and be
    destructively modified.

    Parameters
    ----------
    data: Iterable[NDArray]
        Any sequence of tensors that can be stored in this object. They are
        not checked to have the right number of dimensions. This sequence is
        cloned to avoid nasty side effects when destructively modifying it.
    """

    _data: list[np.ndarray]
    size: int

    def __init__(self, data: Iterable[NDArray]):
        self._data = list(data)
        self.size = len(self._data)

    @overload
    def __getitem__(self, k: int) -> NDArray: ...

    @overload
    def __getitem__(self, k: slice) -> Sequence[NDArray]: ...

    def __getitem__(self, k: Union[int, slice]) -> Union[NDArray, Sequence[NDArray]]:
        #
        # Get MP matrix at position `k`. If 'A' is an MP, we can now
        # do A[k]
        #
        return self._data[k]  # type: ignore

    def __setitem__(self, k: int, value: NDArray) -> NDArray:
        #
        # Replace matrix at position `k` with new tensor `value`. If 'A'
        # is an MP, we can now do A[k] = value
        #
        self._data[k] = value
        return value

    def __iter__(self) -> Iterator[NDArray]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return self.size


from .core import TensorArray
