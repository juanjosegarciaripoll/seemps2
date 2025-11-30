import numpy as np
from seemps.state.array import TensorArray
from .. import tools


class TestTensorArray(tools.TestCase):
    def test_tensor_array_requires_list(self):
        with self.assertRaises(Exception):
            TensorArray()  # type: ignore

    def test_tensor_array_list_is_copied(self):
        data = [np.zeros((1, 2, 1))] * 3
        state = TensorArray(data)
        self.assertTrue(data is not state._data)

    def test_tensor_array_initial_data_is_not_copied(self):
        data = [self.rng.random(size=(1, 2, 1)) for _ in range(3)]
        state = TensorArray(data)
        for A, B in zip(data, state):
            self.assertTrue(A is B)

    def test_tensor_array_can_get_items(self):
        data = [self.rng.normal(size=(1, 2, 1)) for _ in range(3)]
        state = TensorArray(data.copy())
        self.assertTrue(state[0] is data[0])
        self.assertTrue(state[1] is data[1])
        self.assertTrue(state[2] is data[2])

    def test_tensor_array_getitem_accepts_negative_indices(self):
        data = [self.rng.normal(size=(1, 2, 1)) for _ in range(3)]
        state = TensorArray(data.copy())
        self.assertTrue(state[-3] is data[0])
        self.assertTrue(state[-2] is data[1])
        self.assertTrue(state[-1] is data[2])

    def test_tensor_array_get_item_checks_index_bounds(self):
        state = TensorArray([np.zeros((1, 2, 1))] * 3)
        with self.assertRaises(Exception):
            a = state[3]  # type: ignore # noqa: F841
        with self.assertRaises(Exception):
            a = state[-4]  # type: ignore # noqa: F841

    def test_tensor_array_can_set_items(self):
        data = [self.rng.normal(size=(1, 2, 1)) for _ in range(3)]
        new_data = [self.rng.normal(size=(1, 2, 1)) for _ in range(3)]
        state = TensorArray(data.copy())
        state[0] = new_data[0]
        self.assertTrue(state[0] is new_data[0])
        self.assertTrue(state[1] is data[1])
        self.assertTrue(state[2] is data[2])

        state = TensorArray(data.copy())
        state[1] = new_data[1]
        self.assertTrue(state[0] is data[0])
        self.assertTrue(state[1] is new_data[1])
        self.assertTrue(state[2] is data[2])

        state = TensorArray(data.copy())
        state[2] = new_data[2]
        self.assertTrue(state[0] is data[0])
        self.assertTrue(state[1] is data[1])
        self.assertTrue(state[2] is new_data[2])

    def test_tensor_array_set_item_accepts_negative(self):
        data = [self.rng.normal(size=(1, 2, 1)) for _ in range(3)]
        new_data = [self.rng.normal(size=(1, 2, 1)) for _ in range(3)]
        state = TensorArray(data.copy())
        state[0] = new_data[-3]
        self.assertTrue(state[0] is new_data[0])
        self.assertTrue(state[1] is data[1])
        self.assertTrue(state[2] is data[2])

        state = TensorArray(data.copy())
        state[1] = new_data[-2]
        self.assertTrue(state[0] is data[0])
        self.assertTrue(state[1] is new_data[1])
        self.assertTrue(state[2] is data[2])

        state = TensorArray(data.copy())
        state[2] = new_data[-1]
        self.assertTrue(state[0] is data[0])
        self.assertTrue(state[1] is data[1])
        self.assertTrue(state[2] is new_data[2])

    def test_tensor_array_set_item_checks_index_bounds(self):
        state = TensorArray([np.zeros((1, 2, 1))] * 3)
        with self.assertRaises(Exception):
            state[3] = np.zeros((1, 2, 1))
        with self.assertRaises(Exception):
            state[-4] = np.zeros((1, 2, 1))

    def test_tensor_array_rejects_non_array(self):
        if "c++" in self.seemps_version:
            state = TensorArray([np.zeros((1, 2, 1))] * 3)
            with self.assertRaises(Exception):
                state[0] = 1  # type: ignore
            with self.assertRaises(Exception):
                state[0] = [1, 2, 3]  # type: ignore

    def test_tensor_array_length(self):
        state = TensorArray([np.zeros((1, 2, 1))] * 10)
        self.assertEqual(len(state), 10)

    def test_tensor_array_iteration(self):
        data = [self.rng.random(size=(1, 2, 1)) for _ in range(13)]
        state = TensorArray(data.copy())
        for A, B in zip(data, state):
            self.assertTrue(A is B)

    def test_tensor_array_data_property(self):
        data = [self.rng.random(size=(1, 2, 1)) for _ in range(13)]
        state = TensorArray(data)
        self.assertTrue(state._data is not data)
        state._data = data.copy()
        self.assertTrue(state._data is not data)
        for A, B in zip(data, state):
            self.assertTrue(A is B)
