cimport cython
from scipy.linalg.cython_blas cimport dgemm, zgemm

ctypedef fused _evaluation_scalar:
    double
    double complex


def _evaluate_mps(object mps, object mps_indices) -> cnp.ndarray:
    """Evaluate an open-boundary MPS at one index or a batch of indices.

    Row q of `mps_indices` selects the tensor entry
    A[0][0, i[q, 0], :] @ A[1][:, i[q, 1], :] @ ... @ A[-1][:, i[q, -1], 0],
    with negative indices counted from the end, as in NumPy. The entries are
    computed in double precision and returned in the common dtype of the cores.
    """
    cores = list(mps)
    if not cores:
        raise ValueError("Cannot evaluate an empty MPS")
    indices = np.asarray(mps_indices)
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
    if indices.ndim != 2 or indices.shape[1] != len(cores):
        raise ValueError("Expected one index per MPS site")
    if indices.dtype.kind not in "iu":
        raise TypeError("MPS indices must be integers")
    dtype = np.result_type(*cores)
    if dtype.kind not in "fc":
        raise TypeError("MPS evaluation requires floating-point cores")
    if (
        any(A.ndim != 3 for A in cores)
        or cores[0].shape[0] != 1
        or cores[len(cores) - 1].shape[2] != 1
        or any(A.shape[2] != B.shape[0] for A, B in zip(cores, cores[1:]))
    ):
        raise ValueError("Incompatible MPS core shapes")
    dimensions = np.array([A.shape[1] for A in cores])
    if np.any(indices < -dimensions) or np.any(indices >= dimensions):
        raise IndexError("MPS index out of bounds")
    indices = np.ascontiguousarray(
        np.where(indices < 0, indices + dimensions, indices), dtype=np.intp
    )

    computation_dtype = np.complex128 if dtype.kind == "c" else np.float64
    samples = len(indices)
    max_rank = max(A.shape[2] for A in cores)
    order = np.empty(samples, dtype=np.intp)
    offsets = np.empty(dimensions.max() + 1, dtype=np.intp)
    cursor = np.empty(dimensions.max(), dtype=np.intp)
    packed = np.empty(samples * max_rank, dtype=computation_dtype)
    product = np.empty(samples * max_rank, dtype=computation_dtype)
    # Row q of `left` is the product of the first cores, for sample q.
    left = np.ascontiguousarray(cores[0][0, indices[:, 0], :], dtype=computation_dtype)
    for site in range(1, len(cores)):
        core = np.ascontiguousarray(cores[site], dtype=computation_dtype)
        right = np.empty((samples, core.shape[2]), dtype=computation_dtype)
        if computation_dtype is np.float64:
            _evaluation_site[cython.double](
                left, core, indices, site, right, order, offsets, cursor, packed, product
            )
        else:
            _evaluation_site[cython.doublecomplex](
                left, core, indices, site, right, order, offsets, cursor, packed, product
            )
        left = right
    return left[:, 0].astype(dtype, copy=False)


cdef void _evaluation_site(
    _evaluation_scalar[:, ::1] left, _evaluation_scalar[:, :, ::1] core,
    cnp.intp_t[:, ::1] indices, Py_ssize_t site, _evaluation_scalar[:, ::1] right,
    cnp.intp_t[::1] order, cnp.intp_t[::1] offsets, cnp.intp_t[::1] cursor,
    _evaluation_scalar[::1] packed, _evaluation_scalar[::1] product,
) noexcept nogil:
    """Set right[q] = left[q] @ core[:, indices[q, site], :] for all samples q.

    Samples are grouped by their physical index s, so that each group needs a
    single matrix product, packed @ core[:, s, :], where `packed` gathers the
    rows of `left` in the group, and `product` holds the result.
    """
    cdef:
        Py_ssize_t samples = left.shape[0], physical = core.shape[1]
        Py_ssize_t rank_left = core.shape[0], rank_right = core.shape[2]
        Py_ssize_t q, s, g, a, b, count
    # Counting sort: the samples with physical index s are
    # order[offsets[s]:offsets[s + 1]], in their original order.
    for s in range(physical + 1):
        offsets[s] = 0
    for q in range(samples):
        offsets[indices[q, site] + 1] += 1
    for s in range(physical):
        offsets[s + 1] += offsets[s]
    for s in range(physical):
        cursor[s] = offsets[s]
    for q in range(samples):
        s = indices[q, site]
        order[cursor[s]] = q
        cursor[s] += 1

    for s in range(physical):
        count = offsets[s + 1] - offsets[s]
        if count == 0:
            continue
        for g in range(count):
            q = order[offsets[s] + g]
            for a in range(rank_left):
                packed[g * rank_left + a] = left[q, a]
        _evaluation_matmul(
            count, rank_right, rank_left,
            &packed[0], rank_left,
            &core[0, s, 0], physical * rank_right,
            &product[0], rank_right,
        )
        for g in range(count):
            q = order[offsets[s] + g]
            for b in range(rank_right):
                right[q, b] = product[g * rank_right + b]


cdef void _evaluation_matmul(
    int m, int n, int k,
    _evaluation_scalar* A, int lda,
    _evaluation_scalar* B, int ldb,
    _evaluation_scalar* C, int ldc,
) noexcept nogil:
    """C = A @ B for C-ordered matrices with row strides lda, ldb and ldc.

    BLAS uses column-major matrices, so this computes C.T = B.T @ A.T.
    """
    cdef:
        char no_transpose = b"N"
        _evaluation_scalar one = 1, zero = 0
    if _evaluation_scalar is double:
        dgemm(&no_transpose, &no_transpose, &n, &m, &k, &one, B, &ldb, A, &lda, &zero, C, &ldc)
    else:
        zgemm(&no_transpose, &no_transpose, &n, &m, &k, &one, B, &ldb, A, &lda, &zero, C, &ldc)
