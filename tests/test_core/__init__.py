import os

if os.environ.get("SEEMPS_TEST_BACKEND", "off").lower() != "on":
    from . import test_strategy
    from . import test_svd
    from . import test_two_site_split
    from . import test_canonicalize

    __all__ = [
        "test_strategy",
        "test_svd",
        "test_two_site_split",
        "test_canonicalize",
    ]
