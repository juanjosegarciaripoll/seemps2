from typing import Union, Any
import unittest
import h5py
import seemps
from .tools import *
import os


class TestHDF5(MPSTestCase):
    filename = "test_hdf5.hdf5"

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.unlink(self.filename)
        return super().tearDown()

    def test_hdf5_mps_attributes(self):
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mps(file, "M", seemps.random_mps(2, 3))
        with h5py.File(self.filename, "r") as file:
            g = file["M"]
            attrs = g.attrs
            self.assertEqual(len(attrs), 2)
            self.assertEqual(attrs["type"], "MPS")
            self.assertEqual(attrs["version"], 1)

    def test_can_write_complex_mps_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        aux = seemps.state.random(2, 3, 2)
        for i in range(len(aux)):
            aux[i] = aux[i] * 1j
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mps(file, "M", aux)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPS[0]"], aux[0])
        self.assertSimilar(hdf5_data["/M/MPS[1]"], aux[1])
        self.assertSimilar(hdf5_data["/M/MPS[2]"], aux[2])
        self.assertEqual(hdf5_data["/M/MPS[0]"].dtype, aux[0].dtype)
        self.assertEqual(hdf5_data["/M/MPS[1]"].dtype, aux[1].dtype)
        self.assertEqual(hdf5_data["/M/MPS[2]"].dtype, aux[2].dtype)

    def test_can_write_real_mps_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        aux = seemps.state.random(2, 3, 3)
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mps(file, "M", aux)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPS[0]"], aux[0])
        self.assertSimilar(hdf5_data["/M/MPS[1]"], aux[1])
        self.assertSimilar(hdf5_data["/M/MPS[2]"], aux[2])
        self.assertEqual(hdf5_data["/M/MPS[0]"].dtype, aux[0].dtype)
        self.assertEqual(hdf5_data["/M/MPS[1]"].dtype, aux[1].dtype)
        self.assertEqual(hdf5_data["/M/MPS[2]"].dtype, aux[2].dtype)

    def test_can_extend_hdf5(self):
        """Test that a single MPS can be appended to an HDF5 file"""
        self.test_can_write_real_mps_to_hdf5()
        aux = seemps.state.random(2, 4)
        with h5py.File(self.filename, "r+") as file:
            seemps.hdf5.write_mps(file, "X", aux)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4 + 5)
        self.assertEqual(hdf5_data["/X/length"], 4)
        self.assertSimilar(hdf5_data["/X/MPS[0]"], aux[0])
        self.assertSimilar(hdf5_data["/X/MPS[1]"], aux[1])
        self.assertSimilar(hdf5_data["/X/MPS[2]"], aux[2])
        self.assertSimilar(hdf5_data["/X/MPS[3]"], aux[3])

    def test_hdf5_mpo_attributes(self):
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mpo(file, "M", seemps.MPO([np.zeros((1, 2, 2, 1))] * 3))
        with h5py.File(self.filename, "r") as file:
            g = file["M"]
            attrs = g.attrs
            self.assertEqual(len(attrs), 2)
            self.assertEqual(attrs["type"], "MPO")
            self.assertEqual(attrs["version"], 1)

    def test_can_write_complex_mpo_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        aux = seemps.MPO(
            [
                self.rng.normal(size=(10 if i > 0 else 1, 2, 10 if i < 2 else 1))
                * (1.2 + 0.5j)
                for i in range(3)
            ]
        )
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mpo(file, "M", aux)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPO[0]"], aux[0])
        self.assertSimilar(hdf5_data["/M/MPO[1]"], aux[1])
        self.assertSimilar(hdf5_data["/M/MPO[2]"], aux[2])
        self.assertEqual(hdf5_data["/M/MPO[0]"].dtype, aux[0].dtype)
        self.assertEqual(hdf5_data["/M/MPO[1]"].dtype, aux[1].dtype)
        self.assertEqual(hdf5_data["/M/MPO[2]"].dtype, aux[2].dtype)

    def test_can_write_real_mpo_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        aux = seemps.MPO(
            [
                self.rng.normal(size=(10 if i > 0 else 1, 2, 10 if i < 2 else 1))
                for i in range(3)
            ]
        )
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mpo(file, "M", aux)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPO[0]"], aux[0])
        self.assertSimilar(hdf5_data["/M/MPO[1]"], aux[1])
        self.assertSimilar(hdf5_data["/M/MPO[2]"], aux[2])
        self.assertEqual(hdf5_data["/M/MPO[0]"].dtype, aux[0].dtype)
        self.assertEqual(hdf5_data["/M/MPO[1]"].dtype, aux[1].dtype)
        self.assertEqual(hdf5_data["/M/MPO[2]"].dtype, aux[2].dtype)
