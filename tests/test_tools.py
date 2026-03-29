import io
import os
import numpy as np
import scipy.sparse as sp
from seemps import tools
from seemps.tools import (
    InvalidOperation,
    Logger,
    NO_LOGGER,
    annihilation,
    creation,
    log,
    make_logger,
    mkron,
    random_isometry,
    random_Pauli,
)
from .tools import SeeMPSTestCase, almostIsometry, almostIdentity
from contextlib import redirect_stderr


class TestTools(SeeMPSTestCase):
    def test_random_isometry(self):
        for N in range(1, 10):
            for M in range(1, 10):
                A = random_isometry(N, M)
                self.assertTrue(almostIsometry(A))

    def test_random_Pauli(self):
        for _ in range(100):
            σ = random_Pauli()
            self.assertTrue(almostIdentity(σ @ σ))
            self.assertTrue(np.sum(np.abs(σ.T.conj() - σ)) == 0)

    def test_invalid_operation_message_lists_argument_types(self):
        error = InvalidOperation("+", 1, "a")
        self.assertIn("Invalid operation +", str(error))
        self.assertIn("<class 'int'>", str(error))
        self.assertIn("<class 'str'>", str(error))

    def test_make_logger_returns_disabled_logger_when_debug_is_too_low(self):
        old_debug = tools.DEBUG
        try:
            tools.DEBUG = 0
            logger = make_logger(1)
            self.assertIs(logger, NO_LOGGER)
            self.assertIsInstance(logger, Logger)
            self.assertFalse(logger)
        finally:
            tools.DEBUG = old_debug

    def test_make_logger_and_log_emit_output_when_debug_is_enabled(self):
        old_debug = tools.DEBUG
        old_prefix = tools.PREFIX
        try:
            tools.DEBUG = 2
            tools.PREFIX = ""
            stream = io.StringIO()
            with redirect_stderr(stream):
                with make_logger(2) as logger:
                    self.assertTrue(logger)
                    logger("hello")
                log("world", debug_level=2)
            output = stream.getvalue()
            self.assertIn(" hello", output)
            self.assertIn("world", output)
            self.assertEqual(tools.PREFIX, "")
        finally:
            tools.DEBUG = old_debug
            tools.PREFIX = old_prefix

    def test_log_is_silent_when_debug_is_disabled(self):
        old_debug = tools.DEBUG
        try:
            tools.DEBUG = 0
            stream = io.StringIO()
            with redirect_stderr(stream):
                log("hidden")
            self.assertEqual(stream.getvalue(), "")
        finally:
            tools.DEBUG = old_debug

    def test_creation_and_annihilation_are_adjoint(self):
        adag = creation(5)
        a = annihilation(5)
        self.assertSimilar(adag, a.T.conj())
        self.assertSimilar(np.diag(a @ adag), [1, 2, 3, 4, 0])
        self.assertSimilar(np.diag(adag @ a), [0, 1, 2, 3, 4])

    def test_mkron_matches_iterated_kronecker_product(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[0.0, 1.0], [1.0, 0.0]])
        C = np.array([[2.0, 0.0], [0.0, -1.0]])
        out = mkron(A, B, C)
        self.assertFalse(sp.issparse(out))
        self.assertSimilar(out, np.kron(A, np.kron(B, C)))


if "DEBUGSEEMPS" in os.environ:
    from seemps import tools

    tools.DEBUG = int(os.environ["DEBUGSEEMPS"])

__all__ = ["TestTools"]
