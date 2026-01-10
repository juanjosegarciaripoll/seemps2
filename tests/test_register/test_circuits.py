import numpy as np
from ..tools import TestCase
from seemps.register.circuit import interpret_operator


class TestKnownOperators(TestCase):
    def test_interpret_operator_is_case_insensitive(self):
        self.assertSimilar(interpret_operator("Sx"), interpret_operator("SX"))

    def test_interpret_operator_accepts_aliases(self):
        self.assertSimilar(interpret_operator("CX"), interpret_operator("CNOT"))

    def test_interpret_operator_signals_unknown_operators(self):
        with self.assertRaises(Exception):
            interpret_operator("CUp")

    def test_interpret_operator_only_accepts_strings_or_matrices(self):
        interpret_operator("Sz")
        interpret_operator(np.eye(2))
        with self.assertRaises(Exception):
            interpret_operator(1)  # type: ignore
        with self.assertRaises(Exception):
            interpret_operator(np.zeros((3, 1)))  # type: ignore
        with self.assertRaises(Exception):
            interpret_operator(np.zeros((3,)))  # type: ignore
