import numpy as np
from .timing import bench_all
from ncon import ncon  # type: ignore # noqa: F401
from opt_einsum import contract, contract_expression  # type: ignore # noqa: F401


def investigate_two_site_gate_contraction(rng=np.random.default_rng(seed=0x2377312)):
    D = 40
    A = rng.normal(size=(D, 2, D + 1))
    A /= np.linalg.norm(A)
    B = rng.normal(size=(D + 1, 2, D))
    B /= np.linalg.norm(B)
    U = rng.normal(size=(2, 2, 2, 2))
    U2 = U.reshape(4, 4)
    try:
        from ncon import ncon  # type: ignore
    except ImportError:
        pass
    path_info = []
    try:
        from opt_einsum import contract, contract_expression  # type: ignore

        path_info = contract_expression(
            "ijk,klm,nrjl -> inrm", A.shape, B.shape, U.shape, optimize="optimal"
        )
    except ImportError:
        pass

    def method1():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U)

    path = np.einsum_path("ijk,klm,nrjl -> inrm", A, B, U, optimize="optimal")[0]

    def method2():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U, optimize=path)

    def method3():
        a, d, _ = A.shape
        _, e, c = B.shape
        D = d * e
        aux = np.tensordot(A, B, (2, 0)).reshape(a, D, c)
        aux = np.tensordot(U.reshape(D, D), aux, (1, 1)).transpose(1, 0, 2)
        return aux.reshape(a, d, e, c)

    def method4():
        a, d, b = A.shape
        b, e, c = B.shape
        return np.matmul(
            U2, np.matmul(A.reshape(-1, b), B.reshape(b, -1)).reshape(a, -1, c)
        ).reshape(a, d, e, c)

    def method5():
        return contract("ijk,klm,nrjl -> inrm", A, B, U)  # type: ignore

    def method6():
        return path_info(A, B, U)  # type: ignore

    def method7():
        return ncon((A, B, U), ((-1, 2, 1), (1, 3, -4), (-2, -3, 2, 3)))  # type: ignore

    print("---------------\nUnitary evolution contraction")
    bench_all(
        [
            (method1, "einsum", None),
            (method2, "einsum path", None),
            (method3, "tensordot", None),
            (method4, "matmul (library choice)", None),
            (method5, "opt-einsum", None),
            (method6, "opt-einsum path", None),
            (method7, "ncon", None),
        ],
        repeats=1000,
    )
