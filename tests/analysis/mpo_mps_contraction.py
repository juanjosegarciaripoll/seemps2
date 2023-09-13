import numpy as np
from .timing import bench_all
import opt_einsum  # type: ignore
import ncon  # type: ignore


def mpo_left_environment(rng=np.random.default_rng(seed=0x1322312)):
    D = 10
    A = rng.normal(size=(D, 2, D + 3))
    A /= np.linalg.norm(A)
    C = rng.normal(size=(D + 1, 2, 2, D + 4))
    C /= np.linalg.norm(C)
    B = rng.normal(size=(D + 2, 2, D + 5))
    B /= np.linalg.norm(B)
    E = rng.normal(size=(D, D + 1, D + 2))
    E /= np.linalg.norm(E)

    def method1():
        return opt_einsum.contract("acb,ajd,cjie,bif->def", E, A, C, B)

    path = opt_einsum.contract_path(
        "acb,ajd,cjie,bif->def", E, A, C, B, optimize="optimal"
    )
    print(path[0])
    print(path[1])

    method2_proxy = opt_einsum.contract_expression(
        "acb,ajd,cjie,bif->def", E.shape, A.shape, C.shape, B.shape, optimize="optimal"
    )

    def method2():
        return method2_proxy(E, A, C, B)

    def method3():
        # bif,acb->ifac
        aux = np.tensordot(B, E, (0, 2))
        # ifac,cjie->faje
        aux = np.tensordot(aux, C, ([0, 3], [2, 0]))
        # faje,ajd-> def
        aux = np.tensordot(aux, A, ((1, 2), (0, 1))).transpose(2, 1, 0)
        return aux

    # Contraction order:
    # b=1, i=2, c=3, a=4, j=5
    # Output:
    # d=-1, e=-2, f=-3
    def method4():
        return ncon.ncon(
            (B, E, C, A), [(1, 2, -3), (4, 3, 1), (3, 5, 2, -2), (4, 5, -1)]
        )

    print("---------------\nMPO-MPS contractions")
    print(method1().shape)
    print(method3().shape)
    print(method4().shape)
    bench_all(
        [
            (method1, "einsum", None),
            (method2, "einsum path", None),
            (method3, "tensordot (library choice)", None),
            (method4, "ncon", None),
        ],
        repeats=1000,
    )


def mpo_right_environment(rng=np.random.default_rng(seed=0x1322312)):
    D = 10
    A = rng.normal(size=(D, 2, D + 3))
    A /= np.linalg.norm(A)
    C = rng.normal(size=(D + 1, 2, 2, D + 4))
    C /= np.linalg.norm(C)
    B = rng.normal(size=(D + 2, 2, D + 5))
    B /= np.linalg.norm(B)
    E = rng.normal(size=(D + 3, D + 4, D + 5))
    E /= np.linalg.norm(E)

    def method1():
        return opt_einsum.contract("def,ajd,cjie,bif->acb", E, A, C, B)

    path = opt_einsum.contract_path(
        "def,ajd,cjie,bif->acb", E, A, C, B, optimize="optimal"
    )
    print(path[0])
    print(path[1])

    method2_proxy = opt_einsum.contract_expression(
        "def,ajd,cjie,bif->acb", E.shape, A.shape, C.shape, B.shape, optimize="optimal"
    )

    def method2():
        return method2_proxy(E, A, C, B)

    def method3():
        # ajd,def->ajef
        aux = np.tensordot(A, E, (2, 0))
        # ajef,cjie->afci
        aux = np.tensordot(aux, C, ((1, 2), (1, 3)))
        # afci,bif->acb
        aux = np.tensordot(aux, B, ((1, 3), (2, 1)))
        return aux

    # Contraction order:
    # d=1, j=2, e=3, f=4, i=5
    # Output:
    # a=-1, c=-2, b=-3
    def method4():
        return ncon.ncon(
            (A, E, C, B), [(-1, 2, 1), (1, 3, 4), (-2, 2, 5, 3), (-3, 5, 4)]
        )

    print("---------------\nMPO-MPS contractions")
    print(method1().shape)
    print(method3().shape)
    print(method4().shape)
    bench_all(
        [
            (method1, "einsum", None),
            (method2, "einsum path", None),
            (method3, "tensordot (library choice)", None),
            (method4, "ncon", None),
        ],
        repeats=1000,
    )
