import numpy as np

DEFAULT_TOLERANCE = np.finfo(np.float64).eps


def truncate_vector(S, tolerance, max_bond_dimension):
    """Given a vector of Schmidt numbers `S`, a `tolerance` and a maximum
    bond `max_bond_dimension`, determine where to truncate the vector and return
    the absolute error in norm-2 made by that truncation.

    Parameters
    ----------
    S         -- non-negative Schmidt numbers in decreasing order.
    tolerance -- absolute error allowed by the truncation
    max_bond_dimension -- maximum bond dimension (or None)

    Output
    ------
    S         -- new, truncated vector of Schmidt numbers
    error     -- norm-2 error made by the truncation
    """
    if tolerance == 0:
        return S, 0
    # We sum all reduced density matrix eigenvalues, starting from
    # the smallest ones, to avoid rounding errors
    err = np.cumsum(S[::-1] ** 2)
    #
    # This is the sum of all values
    total = err[-1]
    #
    # The vector err[0],err[1],...,err[k] is the error that we make
    # by keeping the singular values from 0 to N-k
    # We find the first position that is above tolerance err[k],
    # which tells us which is the smallest singular value that we
    # have to keep, s[k-1]
    #
    ndx = np.argmax(err >= tolerance * total)
    if max_bond_dimension is not None:
        ndx = max(ndx, S.size - max_bond_dimension)
    #
    # We use that to estimate the size of the array and the actual error
    #
    return S[0 : (S.size - ndx)], err[ndx - 1] if ndx else 0.0
