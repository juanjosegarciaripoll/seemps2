import os

if os.environ.get("SEEMPS_TEST_BACKEND", "off").lower() != "on":
    from . import test_svd
    from . import test_two_site_split

    __all__ = [
        "test_svd",
        "test_two_site_split",
    ]
