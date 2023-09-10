from typing import Any, Callable
from .tools import *
import timeit


def bench_all(all_methods: list[tuple[Callable, str, Any]], repeats=1000):
    method1 = all_methods[0][0]
    for i, (method, name, reorder_output) in enumerate(all_methods):
        n = i + 1
        try:
            t = timeit.timeit(method, number=repeats)
            t = timeit.timeit(method, number=repeats)
        except Exception as e:
            print(e)
            continue
        extra = ""
        if i > 0:
            output = method()
            if reorder_output is not None:
                output = reorder_output(output)
            err = np.linalg.norm(method1() - output)
            extra = f" error={err:1.2g}"
        print(f"Method{n} {name}:\n time={t/repeats:5f}s" + extra)


def investigate_mpo_contraction(rng=np.random.default_rng(seed=0x223775637)):
    A = rng.normal(size=(30, 2, 2, 30))
    A /= np.linalg.norm(A)
    B = rng.normal(size=(10, 2, 13))
    try:
        from ncon import ncon
    except:
        pass
    path_info = []
    try:
        from opt_einsum import contract, contract_expression

        path_info = contract_expression("aijb,cjd->acibd", A.shape, B.shape)
    except:
        pass

    def method1():
        a, i, j, b = A.shape
        c, j, d = B.shape
        return np.einsum("aijb,cjd->acibd", A, B).reshape(a * c, i, b * d)

    def reorder_output(C):
        a, i, j, b = A.shape
        c, j, d = B.shape
        return (
            C.reshape(c, a, i, d, b).transpose(1, 0, 2, 4, 3).reshape(a * c, i, b * d)
        )

    path = np.einsum_path("aijb,cjd->acibd", A, B, optimize="optimal")[0]

    def method2():
        a, i, j, b = A.shape
        c, j, d = B.shape
        return np.einsum("aijb,cjd->acibd", A, B, optimize=path).reshape(
            a * c, i, b * d
        )

    def method3():
        a, i, j, b = A.shape
        c, j, d = B.shape
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
        a, i, j, b = A.shape
        c, j, d = B.shape
        #
        # np.einsum("aijb,cjd->acibd")
        #
        return ncon((A, B), ((-1, -3, 1, -4), (-2, 1, -5))).reshape(c * a, i, d * b)

    def method7():
        a, i, j, b = A.shape
        c, j, d = B.shape
        return contract("aijb,cjd->acibd", A, B).reshape(a * c, i, b * d)

    def method8():
        a, i, j, b = A.shape
        c, j, d = B.shape
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


class TestMPOTensorFold(TestCase):
    def test_contract_A_B(self):
        investigate_mpo_contraction()

        A = self.rng.normal(size=(30, 2, 2, 30))
        B = self.rng.normal(size=(10, 2, 13))

        exact_contraction = np.einsum("cjd,aijb->caidb", B, A).reshape(
            30 * 10, 2, 30 * 13
        )
        fast_contraction = seemps.mpo._mpo_multiply_tensor(A, B)
        self.assertSimilar(exact_contraction, fast_contraction)


def investigate_unitary_contraction(rng=np.random.default_rng(seed=0x2377312)):
    A = rng.normal(size=(10, 2, 13))
    A /= np.linalg.norm(A)
    B = rng.normal(size=(13, 2, 10))
    B /= np.linalg.norm(B)
    U = rng.normal(size=(2, 2, 2, 2))
    U2 = U.reshape(4, 4)
    try:
        from ncon import ncon
    except:
        pass
    path_info = []
    try:
        from opt_einsum import contract, contract_expression

        path_info = contract_expression(
            "ijk,klm,nrjl -> inrm", A.shape, B.shape, U.shape
        )
    except:
        pass

    def method1():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U)

    path = np.einsum_path("ijk,klm,nrjl -> inrm", A, B, U, optimize="optimal")[0]

    def method2():
        return np.einsum("ijk,klm,nrjl -> inrm", A, B, U, optimize=path)

    def method3():
        a, d, b = A.shape
        b, e, c = B.shape
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
        return contract("ijk,klm,nrjl -> inrm", A, B, U)

    def method6():
        return path_info(A, B, U)

    print("---------------\nUnitary evolution contraction")
    bench_all(
        [
            (method1, "einsum", None),
            (method2, "einsum path", None),
            (method3, "tensordot", None),
            (method4, "matmul (library choice)", None),
            (method5, "opt-einsum", None),
            (method6, "opt-einsum path", None),
        ],
        repeats=10000,
    )


class TestTwoSiteEvolutionFold(TestCase):
    def test_contract_U_A_B(self):
        investigate_unitary_contraction()

        A = self.rng.normal(size=(10, 2, 15))
        B = self.rng.normal(size=(15, 3, 13))
        U = self.rng.normal(size=(2 * 3, 2 * 3))

        exact_contraction = np.einsum(
            "ijk,klm,nrjl -> inrm", A, B, U.reshape(2, 3, 2, 3)
        )
        fast_contraction = seemps.state._contractions._contract_nrjl_ijk_klm(U, A, B)
        self.assertSimilar(exact_contraction, fast_contraction)
