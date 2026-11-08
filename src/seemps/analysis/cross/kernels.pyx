"""Native kernels for tensor cross-interpolation."""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, M_PI
from math import prod
from ..mesh import Mesh, RegularInterval, QuantizedInterval, ChebyshevInterval

# TODO: Refactor

ctypedef fused scalar:
    double
    double complex


cdef inline scalar _conj(scalar x) noexcept nogil:
    if scalar is double:
        return x
    else:
        return x.conjugate()


cdef inline double _real(scalar x) noexcept nogil:
    if scalar is double:
        return x
    else:
        return x.real


cdef inline double _abs2(scalar x) noexcept nogil:
    if scalar is double:
        return x * x
    else:
        return x.real * x.real + x.imag * x.imag


cdef object _as_matrix(object A):
    """Return `A` as a C-ordered float64 or complex128 matrix of finite values."""
    A = np.asarray(A)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("Cross decomposition requires a nonempty matrix")
    if A.dtype.kind not in "biufc":
        raise TypeError("Cross decomposition requires real or complex values")
    A = np.ascontiguousarray(A, dtype=np.complex128 if A.dtype.kind == "c" else np.float64)
    if not np.all(np.isfinite(A)):
        raise ValueError("Cross decomposition requires finite values")
    return A


# Cartesian products of multi-indices


def combine_indices(*indices, bint row_major=False):
    """Return the Cartesian product of index blocks, concatenating their columns.

    Each block is a matrix with one multi-index per row. The last block varies
    fastest, or the first one if `row_major` is True. The result has the common
    integer dtype of the blocks, and a single block is returned unchanged.

    Examples
    --------
    >>> combine_indices(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0], [1]]))
    array([[1, 2, 3, 0],
           [1, 2, 3, 1],
           [4, 5, 6, 0],
           [4, 5, 6, 1]])
    """
    cdef Py_ssize_t block, column = 0
    blocks = [np.asarray(a) for a in indices]
    if any(a.ndim != 2 or a.dtype.kind not in "iu" for a in blocks):
        raise ValueError("Index blocks must be two-dimensional integer arrays")
    if len(blocks) == 1:
        return indices[0]
    # In the product, each row of block b is repeated strides[b] times, and
    # the whole block repeats until all rows are covered.
    counts = [len(a) for a in blocks]
    if row_major:
        strides = [prod(counts[:b]) for b in range(len(blocks))]
    else:
        strides = [prod(counts[b + 1 :]) for b in range(len(blocks))]
    result = np.empty((prod(counts), sum(a.shape[1] for a in blocks)), dtype=np.intp)
    if result.size:
        for block in range(len(blocks)):
            block_intp = np.ascontiguousarray(blocks[block], dtype=np.intp)
            _repeat_rows(result, block_intp, column, strides[block])
            column += blocks[block].shape[1]
    return result.astype(np.result_type(*blocks) if blocks else np.intp, copy=False)


cdef void _repeat_rows(cnp.intp_t[:, ::1] result, const cnp.intp_t[:, ::1] block,
                       Py_ssize_t column, Py_ssize_t stride) noexcept nogil:
    """Copy the rows of `block` into result[:, column:column + block.shape[1]],
    repeating each row `stride` times, and the whole block until the end."""
    cdef Py_ssize_t count = block.shape[0], width = block.shape[1]
    cdef Py_ssize_t outer, first, i, j, c
    for outer in range(result.shape[0] // (count * stride)):
        for i in range(count):
            first = (outer * count + i) * stride
            for j in range(stride):
                for c in range(width):
                    result[first + j, column + c] = block[i, c]


# Mesh coordinates


cdef struct _Axis:
    # Nodes of a mesh axis: start + k * step for regular intervals, and
    # shift + step * cos(pi * k / denominator) or
    # shift + step * cos(pi * (2 * k + 1) / denominator) for Chebyshev ones.
    Py_ssize_t size
    bint chebyshev, endpoints
    double start, step, shift, denominator


cdef _Axis _make_axis(object interval):
    cdef _Axis axis
    axis.size = interval.size
    axis.chebyshev = type(interval) is ChebyshevInterval
    axis.endpoints = False
    axis.start = axis.shift = 0
    axis.denominator = 1
    if axis.chebyshev:
        # Same nodes as ChebyshevInterval.__getitem__
        axis.endpoints = interval.endpoints
        axis.denominator = axis.size - 1 if axis.endpoints else 2 * axis.size
        axis.step = (interval.start - interval.stop) / 2
        axis.shift = (interval.start + interval.stop) / 2
    else:
        axis.start, axis.step = interval._start_displaced, interval.step
    return axis


cdef inline Py_ssize_t _index(Py_ssize_t k, Py_ssize_t size) except -1 nogil:
    """Return a valid index into an axis of the given size, wrapping negative ones."""
    if k < 0:
        k += size
    if k < 0 or k >= size:
        with gil:
            raise IndexError("Index out of range")
    return k


cdef inline double _node(_Axis axis, Py_ssize_t k) noexcept nogil:
    """Return the k-th node of the axis."""
    if axis.chebyshev:
        return axis.shift + axis.step * cos(
            M_PI * (k if axis.endpoints else 2 * k + 1) / axis.denominator
        )
    return axis.start + k * axis.step


cdef bint _contributes(cnp.intp_t[:, ::1] points, Py_ssize_t offset, Py_ssize_t count,
                       Py_ssize_t axis) noexcept nogil:
    """Whether any of the rows points[offset:offset+count] has a nonzero index on the axis."""
    cdef Py_ssize_t i
    for i in range(count):
        if points[offset + i, axis] != 0:
            return True
    return False


def indices_to_mesh_coordinates(object mesh, object indices, object map_matrix=None):
    """Return the mesh coordinates of a batch of tensor indices.

    `indices` is either a matrix with one row of tensor indices per sample, or
    a tuple of such matrices ("blocks") representing their Cartesian product, in
    the order of `combine_indices`, which is evaluated without being formed.
    `map_matrix` maps tensor indices to mesh indices, as in `BlackBoxLoadMPS`.

    Returns a (samples, dimension) array, or None for meshes with interval types
    this kernel does not support, for which `Mesh.__getitem__` must be used.
    """
    cdef:
        cnp.intp_t[:, ::1] points
        cnp.intp_t[::1] offsets, counts, strides, index
        double[::1] values
        double[::1, :] output
        _Axis axis
        Py_ssize_t samples, a, block, single, contributing
        Py_ssize_t count, stride, outer, first, i, j, k, q
    if type(mesh) is not Mesh:
        return None
    for interval in mesh.intervals:
        if type(interval) not in (RegularInterval, QuantizedInterval, ChebyshevInterval):
            return None
        if type(interval) is ChebyshevInterval and interval.endpoints and interval.size < 2:
            return None
    blocks = [np.asarray(b) for b in (indices if isinstance(indices, tuple) else (indices,))]
    if any(b.ndim != 2 or b.dtype.kind not in "iu" for b in blocks):
        return None
    dimensions = mesh.dimension
    width = sum(b.shape[1] for b in blocks)
    if width != (dimensions if map_matrix is None else len(map_matrix)):
        raise ValueError("The tensor indices do not match the mesh dimensions")

    # The mesh index of a sample is the sum of the contributions of its blocks.
    # Compute those contributions for each (small) block separately.
    projected = []
    width = 0
    for b in blocks:
        if map_matrix is None:
            contribution = np.zeros((b.shape[0], dimensions), dtype=np.intp)
            contribution[:, width : width + b.shape[1]] = b
        else:
            contribution = b.astype(np.intp) @ np.asarray(
                map_matrix[width : width + b.shape[1]], dtype=np.intp
            )
        projected.append(contribution)
        width += b.shape[1]
    points = np.concatenate(projected)
    # Rows of block b start at points[offsets[b]]. In the Cartesian product,
    # each row of the block is repeated strides[b] times (the last block varies
    # fastest), and the whole block repeats until all samples are covered.
    block_counts = [len(b) for b in blocks]
    offsets = np.array([sum(block_counts[:b]) for b in range(len(blocks))], dtype=np.intp)
    strides = np.array([prod(block_counts[b + 1 :]) for b in range(len(blocks))], dtype=np.intp)
    counts = np.array(block_counts, dtype=np.intp)
    samples = prod(block_counts)

    result = np.empty((samples, dimensions), order="F")
    if samples == 0:
        return result
    output = result
    index = np.empty(samples, dtype=np.intp)
    values = np.empty(max(block_counts))
    for a in range(dimensions):
        axis = _make_axis(mesh.intervals[a])
        with nogil:
            contributing, single = 0, 0
            for block in range(counts.shape[0]):
                if _contributes(points, offsets[block], counts[block], a):
                    contributing += 1
                    single = block
            if contributing <= 1:
                # The coordinates only depend on the rows of one block: compute
                # them once per row, and repeat them over the Cartesian product.
                count, stride = counts[single], strides[single]
                for i in range(count):
                    values[i] = _node(axis, _index(points[offsets[single] + i, a], axis.size))
                for outer in range(samples // (count * stride)):
                    for i in range(count):
                        first = (outer * count + i) * stride
                        for j in range(stride):
                            output[first + j, a] = values[i]
            else:
                for q in range(samples):
                    index[q] = 0
                for block in range(counts.shape[0]):
                    if _contributes(points, offsets[block], counts[block], a):
                        count, stride = counts[block], strides[block]
                        for outer in range(samples // (count * stride)):
                            for i in range(count):
                                k = points[offsets[block] + i, a]
                                first = (outer * count + i) * stride
                                for j in range(stride):
                                    index[first + j] += k
                for q in range(samples):
                    output[q, a] = _node(axis, _index(index[q], axis.size))
    return result


# Maximum volume submatrices


def maxvol_square(object A, int max_iter=10, double tol=1.05) -> tuple:
    """Select a square submatrix of quasi-maximum volume from a tall matrix.

    Return row indices I and coefficients B such that A = B @ A[I, :].
    At most max_iter swaps are performed, stopping when abs(B).max() <= tol.
    If A has no more rows than columns, select every row and return identity B.
    """
    return _maxvol(A, 0, 0, max_iter, tol, tol)


def maxvol_rectangular(object A, tuple rank_kick=(0, 1), int max_iter=10,
                       double tol=1.05, double tol_rect=1.05) -> tuple:
    """Select rows by rectangular maxvol and return A = B @ A[I, :].

    rank_kick bounds the number of rows added after square maxvol. Unselected
    coefficient rows are tested against the Euclidean norm bound tol_rect.
    Selected rows of B form an identity matrix. Supports real and complex A.
    """
    min_kick, max_kick = rank_kick
    return _maxvol(A, min_kick, max_kick, max_iter, tol, tol_rect)


cdef tuple _maxvol(object A, Py_ssize_t min_kick, Py_ssize_t max_kick, int max_iter,
                   double tol, double tol_rect):
    A = _as_matrix(A)
    n, r = A.shape
    if max_iter < 0 or not np.isfinite(tol) or tol < 1:
        raise ValueError("max_iter must be nonnegative and tol must be finite and >= 1")
    if min_kick < 0 or max_kick < min_kick or not np.isfinite(tol_rect) or tol_rect < 0:
        raise ValueError("Invalid rank kick or rectangular tolerance")
    if n <= r:
        return np.arange(n), np.eye(n)
    capacity = min(n, r + max_kick)
    if r + min_kick > capacity:
        raise ValueError("Minimum requested rank exceeds the number of rows")
    B = np.zeros((n, capacity), dtype=A.dtype)
    B[:, :r] = A
    I = np.empty(capacity, dtype=np.intp)
    row, product, norms = np.empty(capacity, A.dtype), np.empty(n, A.dtype), np.empty(n)
    if A.dtype == np.float64:
        rank = _maxvol_core[cython.double](
            B, I, r, max_iter, tol, r + min_kick, tol_rect, row, product, norms
        )
    else:
        rank = _maxvol_core[cython.doublecomplex](
            B, I, r, max_iter, tol, r + min_kick, tol_rect, row, product, norms
        )
    if rank < 0:
        raise np.linalg.LinAlgError("Maxvol requires full column rank")
    return I[:rank], B[:, :rank]


cdef Py_ssize_t _maxvol_core(scalar[:, ::1] B, cnp.intp_t[::1] I, Py_ssize_t r,
                             int max_iter, double tol, Py_ssize_t min_rank, double tol_rect,
                             scalar[::1] row, scalar[::1] product,
                             double[::1] norms) noexcept nogil:
    """Maxvol on a tall matrix A, given as B[:, :r] = A.

    Selects up to B.shape[1] rows I of A, and computes the coefficients B of
    all rows in terms of them, A = B @ A[I], with B[I] the identity matrix.
    Returns the number of selected rows, or -1 if A does not have full rank.
    """
    cdef Py_ssize_t a, b, rank = r
    if not _maxvol_initial_rows(B, I, r, row):
        return -1
    _maxvol_swap_rows(B, I, r, max_iter, tol, row)
    if B.shape[1] > r:
        rank = _maxvol_add_rows(B, I, r, min_rank, tol_rect, row, product, norms)
    for a in range(rank):
        for b in range(rank):
            B[I[a], b] = 1 if a == b else 0
    return rank


cdef bint _maxvol_initial_rows(scalar[:, ::1] B, cnp.intp_t[::1] I, Py_ssize_t r,
                               scalar[::1] row) noexcept nogil:
    """Gauss-Jordan elimination with partial pivoting on the columns of B[:, :r].

    Step k selects as I[k] the row with the largest entry in column k, and uses
    column operations to turn that row into the k-th unit vector. This selects
    the same rows as LU with partial pivoting, and leaves B = A @ inv(A[I]).
    Returns False if A does not have full column rank.
    """
    cdef:
        Py_ssize_t n = B.shape[0], a, c, i, k
        double largest
        scalar pivot, factor
    for k in range(r):
        largest, i = 0, 0
        for a in range(n):
            if abs(B[a, k]) > largest:
                largest, i = abs(B[a, k]), a
        if largest == 0:
            return False
        I[k] = i
        # Column k /= B[i, k], then column c -= B[i, c] * column k for c != k
        pivot = B[i, k]
        for c in range(r):
            row[c] = B[i, c]
        row[k] = 0
        for a in range(n):
            factor = B[a, k] / pivot
            B[a, k] = factor
            for c in range(r):
                B[a, c] -= factor * row[c]
    return True


cdef void _maxvol_swap_rows(scalar[:, ::1] B, cnp.intp_t[::1] I, Py_ssize_t r,
                            int max_iter, double tol, scalar[::1] row) noexcept nogil:
    """Square maxvol on the coefficients B[:, :r] = A @ inv(A[I[:r]]).

    Each step replaces the selected row I[j] with the row i that has the
    largest coefficient |B[i, j]|, which multiplies the volume of A[I] by that
    coefficient, and updates B with a rank-one correction. It stops when no
    coefficient exceeds tol, or after max_iter steps.
    """
    cdef:
        Py_ssize_t n = B.shape[0], a, b, i = 0, j = 0
        int iteration
        double largest
        scalar pivot, factor
    for iteration in range(max_iter):
        largest = 0
        for a in range(n):
            for b in range(r):
                if abs(B[a, b]) > largest:
                    largest, i, j = abs(B[a, b]), a, b
        if largest <= tol:
            break
        I[j] = i
        # B -= B[:, j] (B[i, :] - e_j) / B[i, j]
        pivot = B[i, j]
        for b in range(r):
            row[b] = B[i, b]
        row[j] -= 1
        for a in range(n):
            factor = B[a, j] / pivot
            for b in range(r):
                B[a, b] -= factor * row[b]


cdef Py_ssize_t _maxvol_add_rows(scalar[:, ::1] B, cnp.intp_t[::1] I, Py_ssize_t r,
                                 Py_ssize_t min_rank, double tol_rect, scalar[::1] row,
                                 scalar[::1] product, double[::1] norms) noexcept nogil:
    """Rectangular maxvol, extending the rows I[:r] and coefficients B[:, :r].

    Rows are added while fewer than min_rank are selected, or while some row
    has coefficients with Euclidean norm above tol_rect, up to B.shape[1] rows.
    Each step adds the row i with the largest coefficients and updates B to the
    pseudoinverse coefficients of the new rows with a rank-one correction.
    Returns the number of selected rows.
    """
    cdef:
        Py_ssize_t n = B.shape[0], a, b, i, k = r
        double largest, scale
        scalar factor
    # Squared norms of the coefficients, with -1 marking selected rows.
    for a in range(n):
        norms[a] = 0
        for b in range(r):
            norms[a] += _abs2(B[a, b])
    for a in range(r):
        norms[I[a]] = -1
    while k < B.shape[1]:
        largest, i = -1, 0
        for a in range(n):
            if norms[a] > largest:
                largest, i = norms[a], a
        if k >= min_rank and largest <= tol_rect * tol_rect:
            break
        I[k] = i
        # With v = B @ B[i]^H, B becomes [B - v B[i] / (1 + v[i]), v / (1 + v[i])]
        for b in range(k):
            row[b] = B[i, b]
        for a in range(n):
            product[a] = 0
            for b in range(k):
                product[a] += B[a, b] * _conj(row[b])
        scale = 1 / (1 + _real(product[i]))
        norms[i] = -1
        for a in range(n):
            factor = scale * product[a]
            for b in range(k):
                B[a, b] -= factor * row[b]
            B[a, k] = factor
            if norms[a] >= 0:
                norms[a] = max(0.0, norms[a] - scale * _abs2(product[a]))
        k += 1
    return k


# Partial rank-revealing LU decomposition


def prrlu(object A, double tolerance, int max_rank) -> tuple:
    """Return pivot indices I, J and factors A approximately equal to L @ U.

    L[I] is unit lower triangular and U[:, J] is upper triangular. Eliminate
    the largest Schur-complement entry until tolerance or max_rank is reached,
    retaining at least one pivot. Zero matrices have a rank-one zero
    factorization. Inputs are promoted to float64 or complex128.
    """
    LU = np.array(_as_matrix(A))
    if max_rank < 1 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("PRRLU requires positive max_rank and finite nonnegative tolerance")
    m, n = LU.shape
    limit = min(m, n, max_rank)
    I, J = np.empty(m, dtype=np.intp), np.empty(n, dtype=np.intp)
    L, U = np.zeros((m, limit), LU.dtype), np.zeros((limit, n), LU.dtype)
    if LU.dtype == np.float64:
        rank = _prrlu[cython.double](LU, I, J, L, U, tolerance)
    else:
        rank = _prrlu[cython.doublecomplex](LU, I, J, L, U, tolerance)
    if rank == 0:
        return np.array([0]), np.array([0]), np.ones((m, 1), LU.dtype), np.zeros((1, n), LU.dtype)
    return I[:rank], J[:rank], L[:, :rank], U[:rank]


cdef Py_ssize_t _prrlu(scalar[:, ::1] A, cnp.intp_t[::1] I, cnp.intp_t[::1] J,
                       scalar[:, ::1] L, scalar[:, ::1] U, double tolerance) noexcept nogil:
    """Factor A in place, then copy the factors into the zero matrices L and U,
    undoing the row and column permutations. Returns the rank."""
    cdef Py_ssize_t m = A.shape[0], n = A.shape[1], a, b, p
    cdef Py_ssize_t rank = _prrlu_eliminate(A, I, J, L.shape[1], tolerance)
    for a in range(m):
        for p in range(min(a, rank)):
            L[I[a], p] = A[a, p]
        if a < rank:
            L[I[a], a] = 1
    for p in range(rank):
        for b in range(p, n):
            U[p, J[b]] = A[p, b]
    return rank


cdef Py_ssize_t _prrlu_eliminate(scalar[:, ::1] A, cnp.intp_t[::1] I, cnp.intp_t[::1] J,
                                 Py_ssize_t limit, double tolerance) noexcept nogil:
    """Gaussian elimination with complete pivoting, in place.

    Step p moves the largest entry of the remaining submatrix A[p:, p:] to
    A[p, p], swapping rows and columns (recorded in I and J), and replaces
    A[p+1:, p+1:] by its Schur complement. A ends up with the L factor below
    the diagonal and the U factor on and above it. Elimination stops after
    `limit` steps, or when the largest remaining entry is zero or, after the
    first step, not above `tolerance`. Returns the number of steps.
    """
    cdef:
        Py_ssize_t m = A.shape[0], n = A.shape[1], a, b, i = 0, j = 0, p, rank = 0
        double largest = 0, row_largest
        scalar pivot, factor
    for a in range(m):
        I[a] = a
    for b in range(n):
        J[b] = b
    for a in range(m):
        for b in range(n):
            if abs(A[a, b]) > largest:
                largest, i, j = abs(A[a, b]), a, b
    for p in range(limit):
        if largest == 0 or (p > 0 and largest <= tolerance):
            break
        I[p], I[i] = I[i], I[p]
        J[p], J[j] = J[j], J[p]
        for b in range(n):
            A[p, b], A[i, b] = A[i, b], A[p, b]
        for a in range(m):
            A[a, p], A[a, j] = A[a, j], A[a, p]
        # Schur complement. Track the largest entry of each updated row to
        # choose the row i of the next pivot.
        pivot = A[p, p]
        largest = 0
        for a in range(p + 1, m):
            factor = A[a, p] / pivot
            A[a, p] = factor
            row_largest = 0
            for b in range(p + 1, n):
                A[a, b] -= factor * A[p, b]
                row_largest = max(row_largest, abs(A[a, b]))
            if row_largest > largest:
                largest, i = row_largest, a
        # Then search row i for the column j of the next pivot. Do not compare
        # with row_largest for equality: with fast math, abs() of complex
        # numbers may be computed differently in each loop.
        if largest > 0:
            largest = 0
            for b in range(p + 1, n):
                if abs(A[i, b]) > largest:
                    largest, j = abs(A[i, b]), b
        rank = p + 1
    return rank
