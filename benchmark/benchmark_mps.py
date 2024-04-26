import seemps.state
from seemps.state import CanonicalMPS, GHZ, scprod, DEFAULT_STRATEGY, random_uniform_mps
from seemps.truncate import simplify
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
    return (GHZ(size),)


def make_two_ghz(size):
    return make_ghz(size) * 2


def scalar_product(A, B):
    scprod(A, B)


def random_mps(size, rng=np.random.default_rng(0x23211)):
    state = random_uniform_mps(2, size, 5)
    strategy = DEFAULT_STRATEGY
    return (state, strategy)


def canonicalize(state, strategy):
    return CanonicalMPS(state, strategy=strategy)


def simplify(state, strategy):
    return simplify(state, strategy=strategy)


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
            BenchmarkGroup.run(
                name="RMPS",
                items=[
                    ("canonical", canonicalize, random_mps),
                ],
            ),
            BenchmarkGroup.run(
                name="RMPS",
                items=[
                    ("simplify", canonicalize, random_mps),
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
