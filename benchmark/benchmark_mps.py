import seemps.state
from seemps.state.core import _gemm, GemmOrder
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


def make_two_real_matrices(size):
    return (np.random.normal(size=(size, size)), np.random.normal(size=(size, size)))


def numpy_mmult(A, B):
    return np.matmul(A, B)


def seemps_mmult(A, B):
    return _gemm(A, GemmOrder.NORMAL, B, GemmOrder.NORMAL)


def run_all():
    warmup(64 * 10 * 2 * 10)
    matrix_sizes = [2, 4, 8, 16, 32, 128, 256, 512]
    data = BenchmarkSet(
        name="Numpy",
        environment=system_version(),
        groups=[
            BenchmarkGroup.run(
                name="RMatrix",
                items=[
                    ("matmul", numpy_mmult, make_two_real_matrices, matrix_sizes),
                    ("gemm", seemps_mmult, make_two_real_matrices, matrix_sizes),
                ],
            ),
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
