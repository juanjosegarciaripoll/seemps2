from typing import Any, cast
import pickle
import numpy as np
from ..tools import SeeMPSTestCase

FAILED_TEST_FILE_NAME = "failed_tests.pkl"
LOADED_FAILED_TESTS: dict[str, Any] | None = None
SAVED_FILE_TESTS: dict[str, Any] = {}


class CoreComparisonTestCase(SeeMPSTestCase):
    test_name: str
    test_args: Any

    def _maybe_load_failed_tests(self) -> dict[str, Any]:
        global LOADED_FAILED_TESTS
        if LOADED_FAILED_TESTS is None:
            try:
                with open(FAILED_TEST_FILE_NAME, "rb") as f:
                    LOADED_FAILED_TESTS = pickle.load(f)
            except FileNotFoundError:
                LOADED_FAILED_TESTS = {}
        return cast(dict[str, Any], LOADED_FAILED_TESTS)

    def setUp(self):
        super().setUp()
        failed_tests = self._maybe_load_failed_tests()
        self.test_name = test_name = self.id()
        if test_name in failed_tests:
            self.test_args = failed_tests[test_name]
        else:
            self.test_args = None

    def tearDown(self):
        if hasattr(self, "_outcome"):
            result = self._outcome.result  # type: ignore[attr-defined]
            if result:
                ok = all(test != self for test, _ in result.failures + result.errors)
                if not ok:
                    SAVED_FILE_TESTS[self.test_name] = self.test_args
        return super().tearDown()

    def make_double_arrays(
        self, max_rows: int = 30, max_cols: int = 30, dtype: Any = np.float64
    ) -> list[tuple[int, int, np.ndarray]]:
        """Generate a list of random double arrays for testing."""
        if self.test_args is None:
            self.test_args = [
                (rows, cols, self.rng.normal(size=(rows, cols)).astype(dtype))
                for rows in range(1, max_rows + 1)
                for cols in range(1, max_cols + 1)
                for copies in range(10)
            ]
        return self.test_args

    def make_complex_arrays(
        self, max_rows: int = 30, max_cols: int = 30, dtype: Any = np.complex128
    ) -> list[tuple[int, int, np.ndarray]]:
        """Generate a list of random double arrays for testing."""
        if self.test_args is None:
            self.test_args = [
                (
                    rows,
                    cols,
                    (
                        self.rng.normal(size=(rows, cols))
                        + 1j * self.rng.normal(size=(rows, cols))
                    ).astype(dtype),
                )
                for rows in range(1, max_rows + 1)
                for cols in range(1, max_cols + 1)
                for copies in range(10)
            ]
        return self.test_args
