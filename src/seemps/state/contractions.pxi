cdef cnp.ndarray _as_matrix(cnp.ndarray A, Py_ssize_t rows, Py_ssize_t cols):
    cdef cnp.npy_intp *dims_data = [rows, cols]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 2)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

cdef cnp.ndarray __contract_last_and_first(cnp.ndarray A, cnp.ndarray B):
    """Contract last index of `A` and first from `B`"""
    cdef:
        cnp.ndarray Aarray = <cnp.ndarray>A
        cnp.ndarray Barray = <cnp.ndarray>B
        int ndimA = cnp.PyArray_NDIM(Aarray) - 1
        int ndimB = cnp.PyArray_NDIM(Barray) - 1
        Py_ssize_t Alast = cnp.PyArray_DIM(Aarray, ndimA)
        Py_ssize_t Bfirst = cnp.PyArray_DIM(Barray, 0)
        #
        # By reshaping the two tensors to matrices, we ensure Numpy
        # will always use the CBLAS path (provided the tensors have
        # the same type, of course)
        #
        cnp.ndarray C = __gemm(
            _as_matrix(Aarray, cnp.PyArray_SIZE(Aarray) / Alast, Alast),
            GEMM_NORMAL,
            _as_matrix(Barray, Bfirst, cnp.PyArray_SIZE(Barray) / Bfirst),
            GEMM_NORMAL
        )
        #return <cnp.ndarray>cnp.PyArray_Reshape(C, A.shape[:-1] + B.shape[1:])
        cnp.npy_intp dims_data[32]
    memcpy(dims_data, Aarray.shape, ndimA * sizeof(cnp.npy_intp))
    memcpy(&dims_data[ndimA], Barray.shape+1, ndimB * sizeof(cnp.npy_intp))
    cdef:
        cnp.PyArray_Dims dims = cnp.PyArray_Dims(&dims_data[0], ndimA + ndimB)
    return <cnp.ndarray>cnp.PyArray_Newshape(C, &dims, cnp.NPY_CORDER)

def _contract_last_and_first(A, B) ->  cnp.ndarray:
    """Contract last index of `A` and first from `B`"""
    if (cnp.PyArray_Check(A) == 0 or cnp.PyArray_Check(B) == 0):
        raise ValueError("_contract_last_and_first expects tensors")
    return __contract_last_and_first(A, B)


cdef _matmul = np.matmul

cdef cnp.ndarray _as_2tensor(cnp.ndarray A, Py_ssize_t i, Py_ssize_t j):
    cdef cnp.npy_intp *dims_data = [i, j]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 2)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

cdef cnp.ndarray _as_3tensor(cnp.ndarray A, Py_ssize_t i, Py_ssize_t j, Py_ssize_t k):
    cdef cnp.npy_intp *dims_data = [i, j, k]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 3)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

cdef cnp.ndarray _as_4tensor(cnp.ndarray A, Py_ssize_t i, Py_ssize_t j,
                             Py_ssize_t k, Py_ssize_t l):
    cdef cnp.npy_intp *dims_data = [i, j, k, l]
    cdef cnp.PyArray_Dims dims = cnp.PyArray_Dims(dims_data, 4)
    return <cnp.ndarray>cnp.PyArray_Newshape(A, &dims, cnp.NPY_CORDER)

cdef cnp.ndarray _empty_as_array(cnp.ndarray A):
    return <cnp.ndarray>cnp.PyArray_SimpleNew(cnp.PyArray_NDIM(A),
                                              cnp.PyArray_DIMS(A),
                                              cnp.PyArray_TYPE(A))

cdef cnp.ndarray _empty_matrix(Py_ssize_t rows, Py_ssize_t cols, int dtype):
    cdef cnp.npy_intp *dims_data = [rows, cols]
    return <cnp.ndarray>cnp.PyArray_SimpleNew(2, dims_data, dtype)

cdef cnp.ndarray _empty_vector(Py_ssize_t size, int dtype):
    cdef cnp.npy_intp *dims_data = [size]
    return <cnp.ndarray>cnp.PyArray_SimpleNew(1, dims_data, dtype)

cdef cnp.ndarray _copy_array(cnp.ndarray A):
    return <cnp.ndarray>cnp.PyArray_FROM_OF(A,
                                            cnp.NPY_ARRAY_ENSURECOPY |
                                            cnp.NPY_ARRAY_C_CONTIGUOUS)

cdef cnp.ndarray _resize_matrix(cnp.ndarray A, cnp.npy_intp rows, cnp.npy_intp cols):
    """Equivalent of A[:rows,:cols], creating a fresh new array."""
    cdef:
        cnp.npy_intp old_rows = cnp.PyArray_DIM(A, 0)
        cnp.npy_intp old_cols = cnp.PyArray_DIM(A, 1)
    if rows < 0:
        rows = old_rows
    if cols < 0:
        cols = old_cols
    if rows == old_rows and cols == old_cols:
        return A
    cdef:
        cnp.ndarray output = _empty_matrix(rows, cols, cnp.PyArray_TYPE(A))
        cdef cnp.npy_intp *dims = cnp.PyArray_DIMS(A)
    dims[0] = rows
    dims[1] = cols
    cnp.PyArray_CopyInto(output, A)
    return output

cdef cnp.ndarray _adjoint(cnp.ndarray A):
    cdef cnp.ndarray a = cnp.PyArray_SwapAxes(A, 0, 1)
    if cnp.PyArray_ISCOMPLEX(A):
        a = cnp.PyArray_Conjugate(a, _empty_as_array(a))
    return <cnp.ndarray>a

def _contract_nrjl_ijk_klm(U, A, B) -> cnp.ndarray:
    #
    # Assuming U[n*r,j*l], A[i,j,k] and B[k,l,m]
    # Implements np.einsum('ijk,klm,nrjl -> inrm', A, B, U)
    # See tests.test_contractions for other implementations and timing
    #
    if (cnp.PyArray_Check(A) == 0 or
        cnp.PyArray_Check(B) == 0 or
        cnp.PyArray_Check(U) == 0 or
        cnp.PyArray_NDIM(<cnp.ndarray>A) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>B) != 3 or
        cnp.PyArray_NDIM(<cnp.ndarray>U) != 2):
        raise ValueError("Invalid arguments to _contract_nrjl_ijk_klm")
    cdef:
        # a, d, b = A.shape[0]
        # b, e, c = B.shape[0]
        Py_ssize_t a = cnp.PyArray_DIM(<cnp.ndarray>A, 0)
        Py_ssize_t d = cnp.PyArray_DIM(<cnp.ndarray>A, 1)
        Py_ssize_t b = cnp.PyArray_DIM(<cnp.ndarray>A, 2)
        Py_ssize_t e = cnp.PyArray_DIM(<cnp.ndarray>B, 1)
        Py_ssize_t c = cnp.PyArray_DIM(<cnp.ndarray>B, 2)
    return _as_4tensor(
        _matmul(
            U,
            _as_3tensor(
                <cnp.ndarray>cnp.PyArray_MatrixProduct(
                    _as_matrix(<cnp.ndarray>A, a*d, b),
                    _as_matrix(<cnp.ndarray>B, b, e*c)),
                a, d*e, c)
        ),
        a, d, e, c)
