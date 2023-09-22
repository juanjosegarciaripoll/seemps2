import numpy as np
import scipy.sparse as sp  # type: ignore
from seemps.state import MPS, CanonicalMPS, random_uniform_mps
import seemps.state
import unittest


class TestCase(unittest.TestCase):
    rng = np.random.default_rng(seed=0x1232388472)

    def assertEqualTensors(self, a, b) -> None:
        if not (
            (a.dtype == b.dtype)
            and (a.ndim == b.ndim)
            and (a.shape == b.shape)
            and np.all(a == b)
        ):
            raise AssertionError("Different objects:\na = {a}\nb = {b}")

    def assertSimilar(self, A, B, **kwdargs) -> None:
        if sp.issparse(A):
            A = A.todense()
        elif isinstance(A, MPS):
            A = A.to_vector()
        else:
            A = np.asarray(A)
        if sp.issparse(B):
            B = B.todense()
        elif isinstance(B, MPS):
            B = B.to_vector()
        else:
            B = np.asarray(B)
        if A.ndim != B.ndim or A.shape != B.shape:
            error = ""
        elif np.all(np.isclose(A, B, **kwdargs)):
            return
        else:
            error = f"\nmax(|A-B|)={np.max(np.abs(A - B))}"
        raise self.failureException(f"Objects are not similar:\nA={A}\nB={B}" + error)

    def assertAlmostIdentity(self, A, **kwdargs) -> None:
        if not almostIdentity(A, **kwdargs):
            raise self.failureException(f"Object not close to identity:\nA={A}")

    def random_uniform_mps(
        self, d: int, size: int, truncate: bool = False, **kwdargs
    ) -> CanonicalMPS:
        return CanonicalMPS(
            random_uniform_mps(d, size, truncate=truncate, rng=self.rng), **kwdargs
        )


def similar(A, B, **kwdargs):
    if sp.issparse(A):
        A = A.todense()
    elif isinstance(A, MPS):
        A = A.to_vector()
    if sp.issparse(B):
        B = B.todense()
    elif isinstance(B, MPS):
        B = B.to_vector()
    return (A.shape == B.shape) and np.all(np.isclose(A, B, **kwdargs))


def almostIdentity(L, places=7):
    return np.all(np.isclose(L, np.eye(L.shape[0]), atol=10 ** (-places)))


def almostIsometry(A, places=7):
    N, M = A.shape
    if M < N:
        A = A.T.conj() @ A
    else:
        A = A @ A.T.conj()
    return almostIdentity(A, places=places)


def approximateIsometry(A, direction, places=7):
    if direction > 0:
        a, i, b = A.shape
        A = np.reshape(A, (a * i, b))
        C = A.T.conj() @ A
    else:
        b, i, a = A.shape
        A = np.reshape(A, (b, i * a))
        C = A @ A.T.conj()
    return almostIdentity(C)


def contain_different_objects(A, B):
    return all(a is not b for a, b in zip(A, B))


def contain_same_objects(A, B):
    return all(a is b for a, b in zip(A, B))


def contain_similar_tensors(A, B):
    return all(similar(a, b) for a, b in zip(A, B))


def run_over_random_uniform_mps(function, d=2, N=10, D=10, repeats=10):
    for nqubits in range(1, N + 1):
        for _ in range(repeats):
            function(seemps.state.random(d, N, D))


class MPSTestCase(TestCase):
    def assertSimilarMPS(self, a, b):
        if (a.size != b.size) or not similar(a.to_vector(), b.to_vector()):
            raise AssertionError("Different objects:\na = {a}\nb = {b}")
