import os

if os.environ.get("SEEMPS_TEST_BACKEND", "off").lower() != "on":
    from . import test_svd

    __all__ = [
        "test_svd",
    ]
