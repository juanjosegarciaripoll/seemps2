import unittest
import numpy as np
import scipy.sparse as sp
import seemps
from seemps.state import MPS, CanonicalMPS, MPSSum, random_uniform_mps, random_mps
from seemps.typing import SparseOperator


def identical_lists(l1, l2):
    old_l1 = l1.copy()
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        l1[i] = np.random.normal(1, 2, 1)
        if l1[i] is not l2[i]:
            return False
        if l1[i] is old_l1[i]:
            return False
        if l2[i] is old_l1[i]:
            return False
    return True


class TestCase(unittest.TestCase):
    rng = np.random.default_rng(seed=0x1232388472)
    seemps_version = seemps.version.number

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
            A = A.toarray()  # type: ignore
        elif isinstance(A, MPS) or isinstance(A, MPSSum):
            A = A.to_vector()
        else:
            A = np.asarray(A)
        if sp.issparse(B):
            B = B.toarray()  # type: ignore
        elif isinstance(B, MPS):
            B = B.to_vector()
        else:
            B = np.asarray(B)
        if A.ndim != B.ndim or A.shape != B.shape:
            error = f"They do not have the same shape:\n{A.shape} != {B.shape}"
        elif np.all(np.isclose(A, B, **kwdargs)):
            return
        else:
            error = f"Their difference exceeds their value:\nmax(|A-B|)={np.max(np.abs(A - B))}"
        raise self.failureException(f"Objects are not similar:\nA={A}\nB={B}\n" + error)

    def assertSimilarStates(self, A, B, **kwdargs) -> None:
        if isinstance(A, (MPS, MPSSum)):
            A = A.to_vector()
        else:
            A = np.asarray(A)
        if isinstance(B, (MPS, MPSSum)):
            B = B.to_vector()
        else:
            B = np.asarray(B)
        if len(A) != len(B):
            error = ""
        else:
            u = np.vdot(A, B)
            v = np.linalg.norm(A) * np.linalg.norm(B)
            if np.isclose(np.abs(u), v, **kwdargs):
                return
            error = f"\nmax(|A-B|)={np.linalg.norm(A - B, np.inf)}"
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

    def random_mps(
        self,
        dimensions: list[int],
        truncate: bool = False,
        complex: bool = False,
        **kwdargs,
    ) -> CanonicalMPS:
        return CanonicalMPS(
            random_mps(dimensions, truncate=truncate, rng=self.rng, complex=complex),
            **kwdargs,
        )

    def assertApproximateIsometry(self, A, direction, places=7) -> None:
        if not approximateIsometry(A, direction, places):
            raise self.failureException(f"Tensor is not isometry:\nA={A}")

    def assertSimilarMPS(self, a, b):
        if (a.size != b.size) or not similar(a.to_vector(), b.to_vector()):
            raise AssertionError("Different objects:\na = {a}\nb = {b}")


def similar(A, B, **kwdargs):
    if isinstance(A, SparseOperator):
        A = A.toarray()
    elif isinstance(A, MPS):
        A = A.to_vector()
    if isinstance(B, SparseOperator):
        B = B.toarray()
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


def run_over_random_uniform_mps(function, d=2, N=10, D=10, repeats=10):
    for _ in range(1, N + 1):
        for _ in range(repeats):
            function(seemps.state.random_uniform_mps(d, N, D))
