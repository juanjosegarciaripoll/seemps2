from typing import cast
import numpy as np
import h5py  # type: ignore
import seemps
from seemps.operators import MPO
from seemps.state import random_uniform_mps
from .tools import TestCase
import os


class TestHDF5(TestCase):
    filename = "test_hdf5.hdf5"

    def tearDown(self) -> None:
        if os.path.exists(self.filename):
            os.unlink(self.filename)
        return super().tearDown()

    def test_hdf5_mps_attributes(self):
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mps(file, "M", random_uniform_mps(2, 3))
        with h5py.File(self.filename, "r") as file:
            g = file["M"]
            attrs = g.attrs
            self.assertEqual(len(attrs), 2)
            self.assertEqual(attrs["type"], "MPS")
            self.assertEqual(attrs["version"], 1)

    def test_hdf5_read_whole_file(self):
        with h5py.File(self.filename, "w") as file:
            file.create_dataset("A", data=1)
            file.create_dataset("B", data=2)
            file.create_group("C")
            cast(h5py.Group, file["C"]).create_dataset("D", data=3)
        a = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(a, {"/A": 1, "/B": 2, "/C/D": 3})
        a = seemps.hdf5.read_full_hdf5(self.filename)
        self.assertEqual(a, {"A": 1, "B": 2, "C": {"D": 3, "_attrs": []}, "_attrs": []})

    def test_can_read_and_write_complex_mps_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        state = random_uniform_mps(2, 3, 2)
        for i in range(len(state)):
            state[i] = state[i] * 1j
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mps(file, "M", state)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPS[0]"], state[0])
        self.assertSimilar(hdf5_data["/M/MPS[1]"], state[1])
        self.assertSimilar(hdf5_data["/M/MPS[2]"], state[2])
        self.assertEqual(hdf5_data["/M/MPS[0]"].dtype, state[0].dtype)
        self.assertEqual(hdf5_data["/M/MPS[1]"].dtype, state[1].dtype)
        self.assertEqual(hdf5_data["/M/MPS[2]"].dtype, state[2].dtype)

        with h5py.File(self.filename, "r") as file:
            copy = seemps.hdf5.read_mps(file, "M")
            self.assertEqual(state.size, copy.size)
            self.assertTrue(all(np.all(A == B) for A, B in zip(state, copy)))
            self.assertTrue(all(A.dtype == B.dtype for A, B in zip(state, copy)))

    def test_hdf5_read_mps_signals_errors_for_incorrect_data(self):
        """Test that a single MPS can be written to an HDF5 file"""
        with h5py.File(self.filename, "w") as file:
            file.create_dataset("A", data=2)
            file.create_group("B").create_dataset("C", data=3)
        with h5py.File(self.filename, "r") as file:
            with self.assertRaises(Exception):
                seemps.hdf5.read_mps(file, "A")
            with self.assertRaises(Exception):
                seemps.hdf5.read_mps(file, "B")

    def test_can_read_and_write_real_mps_to_hdf5(self):
        state = random_uniform_mps(2, 3, 3)
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mps(file, "M", state)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPS[0]"], state[0])
        self.assertSimilar(hdf5_data["/M/MPS[1]"], state[1])
        self.assertSimilar(hdf5_data["/M/MPS[2]"], state[2])
        self.assertEqual(hdf5_data["/M/MPS[0]"].dtype, state[0].dtype)
        self.assertEqual(hdf5_data["/M/MPS[1]"].dtype, state[1].dtype)
        self.assertEqual(hdf5_data["/M/MPS[2]"].dtype, state[2].dtype)

        with h5py.File(self.filename, "r") as file:
            copy = seemps.hdf5.read_mps(file, "M")
            self.assertEqual(state.size, copy.size)
            self.assertTrue(all(np.all(A == B) for A, B in zip(state, copy)))
            self.assertTrue(all(A.dtype == B.dtype for A, B in zip(state, copy)))

    def test_can_extend_hdf5(self):
        """Test that a single MPS can be appended to an HDF5 file"""
        self.test_can_read_and_write_real_mps_to_hdf5()
        aux = random_uniform_mps(2, 4)
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
            seemps.hdf5.write_mpo(file, "M", MPO([np.zeros((1, 2, 2, 1))] * 3))
        with h5py.File(self.filename, "r") as file:
            g = file["M"]
            attrs = g.attrs
            self.assertEqual(len(attrs), 2)
            self.assertEqual(attrs["type"], "MPO")
            self.assertEqual(attrs["version"], 1)

    def test_hdf5_read_mpo_signals_errors_for_incorrect_data(self):
        """Test that a single MPS can be written to an HDF5 file"""
        with h5py.File(self.filename, "w") as file:
            file.create_dataset("A", data=2)
            file.create_group("B").create_dataset("C", data=3)
        with h5py.File(self.filename, "r") as file:
            with self.assertRaises(Exception):
                seemps.hdf5.read_mpo(file, "/A")
            with self.assertRaises(Exception):
                seemps.hdf5.read_mpo(file, "/B")

    def test_can_read_and_write_complex_mpo_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        mpo = MPO(
            [
                self.rng.normal(size=(10 if i > 0 else 1, 2, 10 if i < 2 else 1))
                * (1.2 + 0.5j)
                for i in range(3)
            ]
        )
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mpo(file, "M", mpo)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPO[0]"], mpo[0])
        self.assertSimilar(hdf5_data["/M/MPO[1]"], mpo[1])
        self.assertSimilar(hdf5_data["/M/MPO[2]"], mpo[2])
        self.assertEqual(hdf5_data["/M/MPO[0]"].dtype, mpo[0].dtype)
        self.assertEqual(hdf5_data["/M/MPO[1]"].dtype, mpo[1].dtype)
        self.assertEqual(hdf5_data["/M/MPO[2]"].dtype, mpo[2].dtype)

        with h5py.File(self.filename, "r") as file:
            copy = seemps.hdf5.read_mpo(file, "M")
            self.assertEqual(mpo.size, copy.size)
            self.assertTrue(all(np.all(A == B) for A, B in zip(mpo, copy)))
            self.assertTrue(all(A.dtype == B.dtype for A, B in zip(mpo, copy)))

    def test_can_write_real_mpo_to_hdf5(self):
        """Test that a single MPS can be written to an HDF5 file"""
        mpo = MPO(
            [
                self.rng.normal(size=(10 if i > 0 else 1, 2, 10 if i < 2 else 1))
                for i in range(3)
            ]
        )
        with h5py.File(self.filename, "w") as file:
            seemps.hdf5.write_mpo(file, "M", mpo)

        hdf5_data = seemps.hdf5.read_full_hdf5_as_paths(self.filename)
        self.assertEqual(len(hdf5_data), 4)
        self.assertEqual(hdf5_data["/M/length"], 3)
        self.assertSimilar(hdf5_data["/M/MPO[0]"], mpo[0])
        self.assertSimilar(hdf5_data["/M/MPO[1]"], mpo[1])
        self.assertSimilar(hdf5_data["/M/MPO[2]"], mpo[2])
        self.assertEqual(hdf5_data["/M/MPO[0]"].dtype, mpo[0].dtype)
        self.assertEqual(hdf5_data["/M/MPO[1]"].dtype, mpo[1].dtype)
        self.assertEqual(hdf5_data["/M/MPO[2]"].dtype, mpo[2].dtype)

        with h5py.File(self.filename, "r") as file:
            copy = seemps.hdf5.read_mpo(file, "M")
            self.assertEqual(mpo.size, copy.size)
            self.assertTrue(all(np.all(A == B) for A, B in zip(mpo, copy)))
            self.assertTrue(all(A.dtype == B.dtype for A, B in zip(mpo, copy)))
