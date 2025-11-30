from __future__ import annotations
from typing import Any, cast
from .typing import Tensor3, Tensor4
import h5py
from .state import MPS
from .operators import MPO


def _read_hdf5_item_as_path(
    item: h5py.File | h5py.Group | h5py.Dataset, output: list[tuple[str, Any]]
) -> list[tuple[str, Any]]:
    if not isinstance(item, h5py.Dataset):
        for subitem in item.values():
            if isinstance(subitem, h5py.Dataset):
                output.append((cast(str, subitem.name), subitem[()]))
            else:
                _read_hdf5_item_as_path(subitem, output)
    return output


def read_full_hdf5_as_paths(filename: str) -> dict[str, Any]:
    with h5py.File(filename, "r") as file:
        return {key: value for key, value in _read_hdf5_item_as_path(file, [])}


def _read_hdf5_item(item: h5py.File | h5py.Group | h5py.Dataset) -> dict:
    if isinstance(item, h5py.Dataset):
        return item[()]
    output: dict = {key: _read_hdf5_item(subitem) for key, subitem in item.items()}
    output["_attrs"] = list(item.attrs.items())
    return output


def read_full_hdf5(filename: str) -> dict:
    with h5py.File(filename, "r") as file:
        return _read_hdf5_item(file)


def write_mps(parent: h5py.File | h5py.Group, name: str, M: MPS) -> None:
    """Write an MPS to an HDF5 file or group.

    Parameters
    ----------
    parent : h5py.File | h5py.Group
        The file or group where this MPS is created
    name : str
        Name of the subgroup under which the datasets are stored
    M : MPS
        The quantum state to save.

    Examples
    --------
    >>> import h5py
    >>> import seemps.state, seemps.hdf5
    >>> mps = seemps.state.random_uniform_mps(2, 10)
    >>> file = h5py.File("data.hdf5", "w")
    >>> seemps.hdf5.write_mps(file, "state", mps)
    >>> file.close()
    """
    assert isinstance(M, MPS)
    g = parent.create_group(name)
    g.attrs["type"] = "MPS"
    g.attrs["version"] = 1
    g.create_dataset("length", data=len(M))
    for i, A in enumerate(M):
        g.create_dataset(f"MPS[{i}]", shape=A.shape, data=A)


def read_mps(parent: h5py.File | h5py.Group, name: str) -> MPS:
    """Reand an MPS from an HDF5 file or group.

    Parameters
    ----------
    parent : h5py.File | h5py.Group
        The file or group where this MPS is created
    name : str
        Name of the subgroup under which the datasets are stored
    M : MPS
        The quantum state to save.

    Examples
    --------
    >>> import h5py
    >>> import seemps.state, seemps.hdf5
    >>> mps = seemps.state.random_uniform_mps(2, 10)
    >>> file = h5py.File("data.hdf5", "r")
    >>> mps = seemps.hdf5.read_mps(file, "state")
    >>> mps.physical_dimensions()
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> file.close()
    """
    if name in parent:
        g = parent[name]
        if isinstance(g, h5py.Group):
            if g.attrs["type"] == "MPS" and g.attrs["version"] == 1:
                N = cast(int, cast(h5py.Dataset, g["length"])[()])
                # rlim = g["rlim"][()]
                # llim = g["llim"][()]
                return MPS(
                    [
                        cast(Tensor3, cast(h5py.Dataset, g[f"MPS[{i}]"])[()])
                        for i in range(N)
                    ]
                )
    raise Exception(f"Unable to read MPS from HDF5 group {parent}")


# TODO: Add functions to read and write strategies
def write_mpo(parent: h5py.File | h5py.Group, name: str, M: MPO) -> None:
    """Write an MPO to an HDF5 file or group.

    Parameters
    ----------
    parent : h5py.File | h5py.Group
        The file or group where this MPS is created
    name : str
        Name of the subgroup under which the datasets are stored
    M : MPO
        The quantum operator to save.
    """
    assert isinstance(M, MPO)
    g = parent.create_group(name)
    g.attrs["type"] = "MPO"
    g.attrs["version"] = 1
    g.create_dataset("length", data=len(M))
    for i, A in enumerate(M):
        g.create_dataset(f"MPO[{i}]", shape=A.shape, data=A)


def read_mpo(parent: h5py.File | h5py.Group, name: str) -> MPO:
    """Reand an MPO from an HDF5 file or group.

    Parameters
    ----------
    parent : h5py.File | h5py.Group
        The file or group where this MPS is created
    name : str
        Name of the subgroup under which the datasets are stored
    M : MPO
        The quantum state to save.
    """
    if name in parent:
        g = parent[name]
        if (
            isinstance(g, h5py.Group)
            and g.attrs["type"] == "MPO"
            and g.attrs["version"] == 1
        ):
            N = cast(int, cast(h5py.Dataset, g["length"])[()])
            # rlim = g["rlim"][()]
            # llim = g["llim"][()]
            return MPO(
                [
                    cast(Tensor4, cast(h5py.Dataset, g[f"MPO[{i}]"])[()])
                    for i in range(N)
                ]
            )
    raise Exception(f"Unable to read MPO from HDF5 group {parent}")
