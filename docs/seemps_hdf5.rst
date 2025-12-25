.. _seemps_hdf5:

*******************
Reading and writing
*******************

HDF5 is a portable format to store arbitrary numerical data, collected
in a hierarchical structure of nested subgroups, whose leaves are actual
datasets (e.g., arrays, strings, numbers, etc).

This module provides functions to read and write :class:`~seemps.MPS` objects
from and to HDF5 files. Each quantum state is stored in a separate group,
with a name supplied by the user. If `g` is the HDF5 group, the format
consists of the following datasets and attributes:

- `g["length"]`, integer `N` with the size of the MPS.
- `g.attrs["type"]` is `"MPS"`
- `g.attrs["version"]` is 1 for this library.
- `g["MPS[0]"]`, `g["MPS[1]"]` and subsequent fields are datasets for each
  of the tensors.

SeeMPS uses the Python library `h5py <https://www.h5py.org/>`_ to read and write
states in these structured files. For instance, the following code creates a
file with a single MPS stored in the group `"state"`. Afterwards, it reads the
same state from the file.

.. highlight:: python
.. code-block:: python

    import h5py
    import seemps

    # Create a single file, overwriting any existing one
    with h5py.File("data.hdf5", "w") as file:
        seemps.hdf5.write_mps(file, "state" my_mps

    # Read the MPS from the same file, reopening it as read-only
    with h5py.File("data.hdf5", "r") as file
        seemps.hdf5.read_mps(file, "state")

This is a summary of all the functions provided by :py:mod:`seemps.hdf5`.

.. autosummary::

   ~seemps.hdf5.read_mps
   ~seemps.hdf5.write_mps
   ~seemps.hdf5.read_mpo
   ~seemps.hdf5.write_mpo
