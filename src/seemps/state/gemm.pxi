from scipy.linalg.cython_blas cimport dgemm, zgemm

"""
In Python we use C-stype order for arrays. In BLAS, the arrays are in Fortran
order. Thus, a contraction in Python

    C[n,m] = B[n,k] * A[k,m]

is equivalent to *gemm(B', A', C')

    C'[m,n] = A'[m,k] * B'[k,n]

where A', B' and C' are Fotran ordered arrays on the same memory region.
"""

class GemmOrder:
    NORMAL = 0
    TRANSPOSE = 1
    ADJOINT = 2

def _gemm(cnp.ndarray B, int BT, cnp.ndarray A, int AT) -> cnp.ndarray:
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_Check(B) == 0 or
        cnp.PyArray_NDIM(A) != 2 or
        cnp.PyArray_NDIM(B) != 2):
        raise ValueError()
    return __gemm(B, BT, A, AT)

cdef cnp.ndarray __gemm(cnp.ndarray B, int BT, cnp.ndarray A, int AT):
    A = cnp.PyArray_GETCONTIGUOUS(A)
    B = cnp.PyArray_GETCONTIGUOUS(B)
    cdef:
        int Atype = cnp.PyArray_TYPE(A)
        int Btype = cnp.PyArray_TYPE(B)
    if Atype == cnp.NPY_DOUBLE:
        if Btype == cnp.NPY_COMPLEX128:
            return _zgemm(<cnp.ndarray>cnp.PyArray_Cast(A, cnp.NPY_COMPLEX128), AT, B, BT)
        elif Btype == cnp.NPY_DOUBLE:
            return _dgemm(A, AT, B, BT)
        elif Btype == cnp.NPY_COMPLEX64:
            return _zgemm(<cnp.ndarray>cnp.PyArray_Cast(A, cnp.NPY_COMPLEX128), AT,
                          <cnp.ndarray>cnp.PyArray_Cast(B, cnp.NPY_COMPLEX128), BT)
        else:
            return _dgemm(A, AT, <cnp.ndarray>cnp.PyArray_Cast(B, cnp.NPY_DOUBLE), BT)
    elif Atype == cnp.NPY_COMPLEX128:
        if Btype == cnp.NPY_DOUBLE:
            return _zgemm(A, AT, <cnp.ndarray>cnp.PyArray_Cast(B, cnp.NPY_COMPLEX128), BT)
        elif Btype == cnp.NPY_COMPLEX128:
            return _zgemm(A, AT, B, BT)
        elif Btype == cnp.NPY_COMPLEX64:
            return _zgemm(A, AT, <cnp.ndarray>cnp.PyArray_Cast(B, cnp.NPY_COMPLEX128), BT)
        elif Btype == cnp.NPY_DOUBLE:
            return _zgemm(A, AT, <cnp.ndarray>cnp.PyArray_Cast(B, cnp.NPY_COMPLEX128), BT)
    elif Atype == cnp.NPY_COMPLEX64:
        return __gemm(B, BT, <cnp.ndarray>cnp.PyArray_Cast(A, cnp.NPY_COMPLEX128), AT)
    else:
        return __gemm(B, BT, <cnp.ndarray>cnp.PyArray_Cast(A, cnp.NPY_DOUBLE), AT)
    raise ValueError((A.dtype, B.dtype))

cdef cnp.ndarray _dgemm(cnp.ndarray A, int AT, cnp.ndarray B, int BT):
    cdef:
        int m, n, k, lda, ldb
        char *Aorder
        char *Border
    if AT == GEMM_NORMAL:
        m = lda = cnp.PyArray_DIM(A, 1)
        k = cnp.PyArray_DIM(A, 0)
        Aorder = 'N'
    else:
        m = cnp.PyArray_DIM(A, 0)
        k = lda = cnp.PyArray_DIM(A, 1)
        Aorder = 'T'
    if BT == GEMM_NORMAL:
        n = cnp.PyArray_DIM(B, 0)
        ldb = k
        Border = 'N'
    else:
        n = cnp.PyArray_DIM(B, 1)
        ldb = n
        Border = 'T'
    cdef:
        cnp.ndarray C = _empty_matrix(n, m, cnp.NPY_DOUBLE)
        double alpha = 1.0
        double beta = 0.0
    dgemm(Aorder, Border, &m, &n, &k, &alpha,
          <double*>cnp.PyArray_DATA(A), &lda,
          <double*>cnp.PyArray_DATA(B), &ldb,
          &beta,
          <double*>cnp.PyArray_DATA(C), &m)
    return C

cdef cnp.ndarray _zgemm(cnp.ndarray A, int AT, cnp.ndarray B, int BT):
    cdef:
        int m, n, k, lda, ldb
        char *Aorder
        char *Border
    if AT == GEMM_NORMAL:
        m = lda = cnp.PyArray_DIM(A, 1)
        k = cnp.PyArray_DIM(A, 0)
        Aorder = 'N'
    else:
        m = cnp.PyArray_DIM(A, 0)
        k = lda = cnp.PyArray_DIM(A, 1)
        Aorder = 'C' if AT == GEMM_ADJOINT else 'T'
    if BT == GEMM_NORMAL:
        n = cnp.PyArray_DIM(B, 0)
        ldb = k
        Border = 'N'
    else:
        n = cnp.PyArray_DIM(B, 1)
        ldb = n
        Border = 'C' if BT == GEMM_ADJOINT else 'T'
    cdef:
        cnp.ndarray C = _empty_matrix(n, m, cnp.NPY_COMPLEX128)
        double complex alpha = 1.0
        double complex beta = 0.0
    zgemm(Aorder, Border, &m, &n, &k, &alpha,
          <double complex*>cnp.PyArray_DATA(A), &lda,
          <double complex*>cnp.PyArray_DATA(B), &ldb,
          &beta,
          <double complex*>cnp.PyArray_DATA(C), &m)
    return C
