# %%
import numpy as np
from time import perf_counter
import warnings
from seemps.state import Truncation, NO_TRUNCATION, Simplification, CanonicalMPS
from seemps.truncate import SIMPLIFICATION_STRATEGY
from seemps.analysis.sampling import evaluate_mps, random_mps_indices
from seemps.analysis.mesh import Mesh, RegularHalfOpenInterval, mps_to_mesh_matrix
from seemps.analysis.factories import mps_interval, mps_tensor_sum
from seemps.analysis.chebyshev import chebyshev_coefficients, cheb2mps
from seemps.analysis.polynomials import mps_from_polynomial

import seemps.tools

# seemps.tools.DEBUG = 3


# %%
def distance_in_norm_inf(tensor1: np.ndarray, tensor2: np.ndarray) -> float:
    r"""Returns the distance in the $L\infty$ norm between two tensors of the same shape."""
    if tensor1.shape != tensor2.shape:
        raise ValueError("The tensors are of different shape")
    return np.max(np.abs(tensor1 - tensor2))


# %%
def interpolate_gaussian_product(m, n, d, t, order):
    # Define function
    a, b = -1, 1
    func = lambda x: np.exp(-x)
    func_tensor = lambda tensor: func(np.sum(tensor**2, axis=-1))
    COMPUTER_PRECISION = SIMPLIFICATION_STRATEGY.replace(
        tolerance=np.finfo(np.double).eps,
        simplification_tolerance=np.finfo(np.double).eps,
        simplify=Simplification.DO_NOT_SIMPLIFY,
        method=Truncation.RELATIVE_SINGULAR_VALUE,
        normalize=False,
    )
    strategy_cheb = SIMPLIFICATION_STRATEGY.replace(
        tolerance=t,
        simplification_tolerance=t,
        method=Truncation.RELATIVE_SINGULAR_VALUE,
        normalize=False,
    )

    # Perform Chebyshev interpolation
    time_start = perf_counter()
    interval = RegularHalfOpenInterval(a, b, 2**n)
    mps_x_squared = mps_from_polynomial([0, 0, 1], interval, strategy=NO_TRUNCATION)
    mps_domain = mps_tensor_sum(
        [mps_x_squared] * m,
        mps_order=order,
        strategy=COMPUTER_PRECISION,
    )
    (start, stop) = (0, m * b**2)
    coefficients = chebyshev_coefficients(func, d, start, stop)
    mps_cheb = cheb2mps(coefficients, x=mps_domain, strategy=strategy_cheb)
    time_stop = perf_counter()
    time = time_stop - time_start
    # print(mps_cheb.bond_dimensions())

    # Evaluate MPS
    max_bond = max(mps_cheb.bond_dimensions())
    rng = np.random.default_rng(42)
    mps_indices = random_mps_indices(mps_cheb, rng=rng)
    y_mps = evaluate_mps(mps_cheb, mps_indices)
    T = mps_to_mesh_matrix([n] * m, order=order)
    mesh = Mesh([interval] * m)
    mesh_coordinates = mesh[mps_indices @ T]
    y_vec = func_tensor(mesh_coordinates)
    error = distance_in_norm_inf(y_vec, y_mps)

    return time, error, max_bond


# %%
# Raise warnings for wrong state norms
warnings.filterwarnings("error")


# %%
# Benchmark
m = 2
n = 10
d = 20
t = 1e-10
# t = np.finfo(np.double).eps
order = "A"

if True:
    time, error, max_bond = interpolate_gaussian_product(m, n, d, t, order)
    print(
        f"-----\nExpansion completed in {time:5f}s,\n"
        f"with norm-2 error {error:6e}, "
        f"\nand bond dimension {max_bond}"
    )

# %%
# Benchmark
m = 5
n = 10
d = 20
# t = 1e-10
order = "A"
repeats = 20
tot_time = 0.0
for _ in range(repeats):
    time, error, max_bond = interpolate_gaussian_product(m, n, d, t, order)
    tot_time += time
print(
    f"-----\nExpansion completed in {tot_time/repeats:5f}s,\n"
    f"with norm-2 error {error:6e}, "
    f"\nand bond dimension {max_bond}"
)
