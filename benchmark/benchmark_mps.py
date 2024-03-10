import seemps.state
from benchmark import BenchmarkSet, BenchmarkGroup
import numpy as np
import sys

GENERATOR = np.random.default_rng(13221231)


def warmup(size, dtype=np.double):
    for _ in range(10):
        _ = np.empty(size, dtype=dtype)


def system_version():
    v = sys.version_info
    return f"Python {v.major}.{v.minor}.{v.micro} NumPy {np.version.full_version}"


def propagate_size(size):
    return (size,)


def make_ghz(size):
    return (seemps.state.GHZ(size),)


def make_two_ghz(size):
    return make_ghz(size) * 2


def scalar_product(A, B):
    seemps.state.scprod(A, B)


def run_all():
    warmup(64 * 10 * 2 * 10)
    data = BenchmarkSet(
        name="Numpy",
        environment=system_version(),
        groups=[
            BenchmarkGroup.run(
                name="MPS",
                items=[
                    ("ghz", make_ghz, propagate_size),
                ],
            ),
            BenchmarkGroup.run(
                name="RMPS",
                items=[
                    ("scprod", scalar_product, make_two_ghz),
                ],
            ),
            BenchmarkGroup.run(
                name="CMPS",
                items=[
                    ("scprod", scalar_product, make_two_ghz),
                ],
            ),
        ],
    )
    if len(sys.argv) > 1:
        data.write(sys.argv[1])
    else:
        data.write("./benchmark_numpy.json")


if __name__ == "__main__":
    print(sys.argv)
    run_all()
