from typing import Any, cast
import pickle
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
