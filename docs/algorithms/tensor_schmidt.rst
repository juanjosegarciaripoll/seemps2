*********************
Schmidt decomposition
*********************

Let us assume a quantum state with two subsystems, labeled by physical indices
:math:`i` and :math:`j`, running over dimensions :math:`d_1` and :math:`d_2`.
In quantum notation

.. math::
    |\psi\rangle = \sum_{i=1}^{d_i} \sum_{j=1}^{d_j} \psi_{i,j} |i\rangle|j\rangle

This quantum state has a minimal decomposition, called the Schmidt decomposition,
which expresses the state as a convex combination of states in two orthogonal
basis:

.. math::
    |\psi\rangle = \sum_{k=1}^d s_k |\phi_k\rangle|\xi_k\rangle

such that :math:`\langle\phi_k|\phi_j\rangle=\delta_{jk}` and
:math:`\langle\xi_k|\xi_j\rangle=\delta_{jk}`, with non-negative Schmidt
weights $s_k\geq 0$.

From a tensor's perspective, what we have achieved is a decomposition of the
form

.. math::
    \psi_{ij} = \sum_k \phi_{ki} s_k \xi_{kj}

If the number of Schmidt vectors `d` is small, this decomposition is advantageous.
This may be the case also when the weights :math:`|s_k|^2` are small enough that
they can be dropped above a certain size.

How are the Schmidt vectors computed?  The orthogonal basis for the `i` and `j`
subsystems are actually the eigenstates of the reduced density matrices.
For instance

.. math::
    \rho_1 = \sum_k s_k^2 |\phi_k\rangle\langle\phi_k|
           = \mathrm{tr}_2|\psi\rangle\langle\psi|

We could thus compute those matrices, diagonalize them and obtain the tensors
:math:`\phi_{ki}` and :math:`\xi_{kj}`.

However, a more efficient approach is to use the singular-value decomposition
or SVD, provided in Python by :func:`scipy.linalg.svd`. This algorithm recreates
a matrix :math:`A` as the product of two isometries :math:`U` and :math:`V`,
and a diagonal matrix of weights :math:`\Sigma_{ij}=s_i\delta_{ij}`. thus

.. math::
    A = U \Sigma V^T

Using this decomposition

.. math::
    A_{ij} = U_{ik} s_k V_{jk}

Now, if we identify the matrix :math:`A` with our original tensor :math:`\psi`,
we can identify the isometries with the Schmidt basis

.. math::
    \psi_{ij} = U_{ik} s_k V_{jk} = \phi_{ki} s_k \xi_{kj}
