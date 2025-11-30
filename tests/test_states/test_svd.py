import scipy.linalg
from seemps.state.schmidt import _destructive_svd
from .. import tools


class TestSVD(tools.TestCase):
    def size_iterator(self, max_size: int = 6):
        for m in range(1, max_size):
            for n in range(1, max_size):
                yield (m, n)

    def test_real_destructive_svd(self):
        for s in self.size_iterator():
            A = self.rng.normal(size=s)
            U, s, VT = scipy.linalg.svd(A, full_matrices=False, compute_uv=True)
            self.assertSimilar(A, (U * s) @ VT, rtol=1e-10, atol=1e-15)
            U, s, VT = _destructive_svd(A.copy())
            self.assertSimilar(A, (U * s) @ VT, rtol=1e-10, atol=1e-15)

    def test_complex_destructive_svd(self):
        for s in self.size_iterator():
            A = self.rng.normal(size=s) + 1j * self.rng.normal(size=s)
            U, s, VT = scipy.linalg.svd(A, full_matrices=False, compute_uv=True)
            self.assertSimilar(A, (U * s) @ VT, rtol=1e-10, atol=1e-15)
            U, s, VT = _destructive_svd(A.copy())
            self.assertSimilar(A, (U * s) @ VT, rtol=1e-10, atol=1e-15)
