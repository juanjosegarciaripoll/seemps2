import numpy as np
from .timing import bench_all
from ncon import ncon  # type: ignore
from opt_einsum import contract, contract_expression  # type: ignore


def investigate_mpo_contraction(rng=np.random.default_rng(seed=0x223775637)):
    A = rng.normal(size=(30, 2, 2, 30))
    A /= np.linalg.norm(A)
    B = rng.normal(size=(10, 2, 13))
    path_info = []
    path_info = contract_expression(
        "aijb,cjd->acibd", A.shape, B.shape, optimize="optimal"
    )

    def method1():
        a, i, _, b = A.shape
        c, _, d = B.shape
        return np.einsum("aijb,cjd->acibd", A, B).reshape(a * c, i, b * d)

    def reorder_output(C):
        a, i, _, b = A.shape
        c, _, d = B.shape
        return (
            C.reshape(c, a, i, d, b).transpose(1, 0, 2, 4, 3).reshape(a * c, i, b * d)
        )

    path = np.einsum_path("aijb,cjd->acibd", A, B, optimize="optimal")[0]

    def method2():
        a, i, _, b = A.shape
        c, _, d = B.shape
        return np.einsum("aijb,cjd->acibd", A, B, optimize=path).reshape(
            a * c, i, b * d
        )

    def method3():
        a, i, _, b = A.shape
        c, _, d = B.shape
        # tensordot(A, B, (2, 1)) -> (a,i,b,c,d)
        return (
            np.tensordot(A, B, (2, 1)).transpose(0, 3, 1, 2, 4).reshape(a * c, i, b * d)
        )

    def method4():
        a, i, j, b = A.shape
        c, j, d = B.shape
        #
        # Matmul takes two arguments
        #     A(a, 1, i, b, j)
        #     B(1, c, 1, j, d)
        # It broadcasts, repeating the indices that are of size 1
        #     A(a, c, i, b, j)
        #     B(a, c, i, j, d)
        # And then multiplies the matrices that are formed by the last two
        # indices, (b,j) * (j,d) -> (b,d) so that the outcome has size
        #     C(a, c, i, b, d)
        #
        return np.matmul(
            A.transpose(0, 1, 3, 2).reshape(a, 1, i, b, j), B.reshape(1, c, 1, j, d)
        ).reshape(a * c, i, b * d)

    def method5():
        a, i, j, b = A.shape
        c, j, d = B.shape
        #
        # Matmul takes two arguments
        #     B(c, 1, 1, d, j)
        #     A(1, a, i, j, b)
        # It broadcasts, repeating the indices that are of size 1
        #     B(c, a, i, d, j)
        #     A(c, a, i, j, b)
        # And then multiplies the matrices that are formed by the last two
        # indices, (d,j) * (j,b) -> (b,d) so that the outcome has size
        #     C(c, a, i, d, b)
        #
        return np.matmul(
            B.transpose(0, 2, 1).reshape(c, 1, 1, d, j), A.reshape(1, a, i, j, b)
        ).reshape(c * a, i, d * b)

    def method6():
        a, i, _, b = A.shape
        c, _, d = B.shape
        #
        # np.einsum("aijb,cjd->acibd")
        #
        return ncon((A, B), ((-1, -3, 1, -4), (-2, 1, -5))).reshape(c * a, i, d * b)

    def method7():
        a, i, _, b = A.shape
        c, _, d = B.shape
        return contract("aijb,cjd->acibd", A, B).reshape(a * c, i, b * d)

    def method8():
        a, i, _, b = A.shape
        c, _, d = B.shape
        return path_info(A, B).reshape(a * c, i, b * d)

    print("\n----------\nTensor contractions for MPO * MPS")
    bench_all(
        [
            (method1, "einsum", None),
            (method2, "einsum path", None),
            (method3, "tensordot", None),
            (method4, "matmul order1", None),
            (method5, "matmul order2 (library choice)", reorder_output),
            (method6, "ncon", None),
            (method7, "opt-einsum", None),
            (method8, "opt-einsum path", None),
        ],
        repeats=1000,
    )
