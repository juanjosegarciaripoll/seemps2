"""
We want to optimize a DMRG contraction that originally was written as
    a = opt_einsum.contract_path(
        'acb,cikjld,edf,bklf->aije',
        ArrayShaped((100, 120, 100)), # L
        ArrayShaped((120, 2, 2, 2, 2, 120)), # H12
        ArrayShaped((100, 120, 100)), # R
        ArrayShaped((100, 2, 2, 100)), # psi
    )
Here, the L and R tensors are the left and right environment for a
Hamiltonian whose central tensor is H12, and is to be contracted with 'psi'.
"""

import time
from itertools import product, permutations
from opt_einsum import contract_path, contract
from opt_einsum.typing import ArrayShaped
import numpy as np

Dimensions = tuple[int]
Labels = str


def expt_0():
    indices = [
        ((100, 16, 100), "acb"),
        ((16, 4, 4, 16), "cijd"),
        ((100, 16, 100), "edf"),
        ((100, 4, 100), "bjf"),
    ]
    size = {
        letter: d
        for dimensions, string in indices
        for d, letter in zip(dimensions, string)
    }
    print(size)

    output_indices = "aie"
    # 839680000
    best_cost = 1e128
    best_cost_route = None
    best_time = 1e128
    best_time_route = None
    i = 0
    for orders in product(*tuple(permutations(ijk) for _, ijk in indices)):
        arrays = [
            np.random.rand(*tuple(size[letter] for letter in ijk)) for ijk in orders
        ]
        shapes = [ArrayShaped(tuple(size[letter] for letter in ijk)) for ijk in orders]
        output = "".join(
            letter for ijk in orders for letter in ijk if letter in output_indices
        )
        route = ",".join("".join(ijk) for ijk in orders) + "->" + output
        _, contraction = contract_path(
            route, *shapes, optimize="optimal", use_blas=True
        )
        t0 = contraction.opt_cost
        t = time.perf_counter()
        for _ in range(10):
            contract(route, *arrays, optimize="optimal", use_blas=True)
        t = time.perf_counter() - t
        print(f"{route} -> {t0:15} {t:1.5f} ops")
        i += 1
        if t < best_time:
            best_time = t
            best_time_route = route
        if t0 < best_cost:
            best_cost = t0
            best_cost_route = route
    print(f"Best contraction per time:\n{best_time_route} -> {best_time} s")
    print(f"Best contraction:\n{best_cost_route} -> {best_cost} ops")


def expt_1():
    indices = [
        ((50, 16, 50), "acb"),
        ((16, 2, 2, 2, 2, 16), "cikjld"),
        ((50, 16, 50), "edf"),
        ((50, 2, 2, 50), "bklf"),
    ]
    size = {
        letter: d
        for dimensions, string in indices
        for d, letter in zip(dimensions, string)
    }
    print(size)

    output_indices = "aije"

    best_time = 1e128
    best_route = None
    redundancy = 0
    i = 0
    t = 0
    for orders in product(*tuple(permutations(ijk) for _, ijk in indices)):
        shapes = [np.random.rand(*(size[letter] for letter in ijk)) for ijk in orders]
        output = "".join(
            letter for ijk in orders for letter in ijk if letter in output_indices
        )
        route = ",".join("".join(ijk) for ijk in orders) + "->" + output
        print(route)
        print(list(a.shape for a in shapes))
        path_info, _ = np.einsum_path(route, *shapes, optimize="optimal")
        t = time.perf_counter()
        np.einsum(route, *shapes, optimize=path_info)
        t = time.perf_counter() - t
        print(f"{route} -> {t} ops {contraction.speedup:10.0f}")
        i += 1
        if t < best_time:
            best_time = t
            best_route = route
            redundancy = 0
        elif t == best_time:
            redundancy += 1
    print(f"Best contraction:\n{best_route} -> {t} ops")


"""
expt_0()
...
Best contraction per time:
acb,cijd,dfe,fjb->aie -> 0.0780046998988837 s
...
  Complete contraction:  acb,cijd,dfe,fjb->aie
         Naive scaling:  8
     Optimized scaling:  6
      Naive FLOP count:  1.638e+12
  Optimized FLOP count:  3.379e+8
   Theoretical speedup:  4.848e+3
  Largest intermediate:  6.400e+5 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   5           GEMM          fjb,acb->fjac                    cijd,dfe,fjac->aie
   6           TDOT        fjac,cijd->faid                         dfe,faid->aie
   5           TDOT          faid,dfe->aie                              aie->aie
"""

path_info, contraction = contract_path(
    "acb,cijd,dfe,fjb->aie",
    ArrayShaped((100, 16, 100)),
    ArrayShaped((16, 4, 4, 16)),
    ArrayShaped((16, 100, 100)),
    ArrayShaped((100, 4, 100)),
    optimize="optimal",
    use_blas=True,
)
print(path_info)
print(contraction)
