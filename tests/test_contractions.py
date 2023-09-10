from .tools import *
import timeit


def investigate_mpo_contraction():
    A = np.random.randn(30, 2, 2, 30)
    A /= np.linalg.norm(A)
    B = np.random.randn(10, 2, 13)
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
        return np.einsum("aijb,cjd->acibd", A, B).reshape(30 * 10, 2, 30 * 13)

    def reorder_output(C):
        a, i, j, b = A.shape
        c, j, d = B.shape
        return (
            C.reshape(c, a, i, d, b).transpose(1, 0, 2, 4, 3).reshape(a * c, i, b * d)
        )

    path = np.einsum_path("aijb,cjd->acibd", A, B, optimize="optimal")[0]

    def method2():
        return np.einsum("aijb,cjd->acibd", A, B, optimize=path).reshape(
            30 * 10, 2, 30 * 13
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

    all_methods = [
        (method1, "einsum"),
        (method2, "einsum path"),
        (method3, "tensordot"),
        (method4, "matmul order1"),
        (method5, "matmul order2"),
        (method6, "ncon"),
        (method7, "opt-einsum"),
        (method8, "opt-einsum path"),
    ]

    repeats = 1000
    print("\n----------\nTensor contractions for MPO * MPS")
    for i, (method, name) in enumerate(all_methods):
        n = i + 1
        try:
            t = timeit.timeit(method, number=repeats)
            t = timeit.timeit(method, number=repeats)
        except Exception as e:
            print(e)
            continue
        extra = ""
        output = method()
        if n == 5:
            extra += " (<- library_choice)"
            output = reorder_output(output)
        if i > 0:
            err = np.linalg.norm(method1() - output)
            extra = f" error={err:1.2g}" + extra
        print(f"Method{n} {name}:\n time={t/repeats:5f}s" + extra)


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


def investigate_unitary_contraction():
    A = np.random.randn(10, 2, 13)
    A /= np.linalg.norm(A)
    B = np.random.randn(13, 2, 10)
    B /= np.linalg.norm(B)
    U = np.random.randn(2, 2, 2, 2)
    U2 = U.reshape(4, 4)

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

    repeats = 10000
    t = timeit.timeit(method1, number=repeats)
    t = timeit.timeit(method1, number=repeats)
    print("\n----------\nTensor contractions for unitary evolution")
    print(f"Method1 {t/repeats}s")

    t = timeit.timeit(method2, number=repeats)
    t = timeit.timeit(method2, number=repeats)
    print(f"Method2 {t/repeats}s")

    t = timeit.timeit(method3, number=repeats)
    t = timeit.timeit(method3, number=repeats)
    print(f"Method3 {t/repeats}s")

    U2 = U.reshape(4, 4)
    t = timeit.timeit(method4, number=repeats)
    t = timeit.timeit(method4, number=repeats)
    print(f"Method4 {t/repeats}s (<- library choice)")

    for i, m in enumerate([method2, method3, method4]):
        err = np.linalg.norm(method1() - m())
        print(f"Method{i+2} error = {err}")


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
