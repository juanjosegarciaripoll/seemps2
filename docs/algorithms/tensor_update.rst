.. _tensor_updates:

**********
MPS update
**********

There are many types of algorithmic updates that one might want to implement
using matrix-product-state forms. For instance, in orden of complexity:

- Apply a local quantum gate, such as a :math:`\sigma_x` rotation onto a qubit.
- Apply an entanging gate onto a pair of neighboring qubits.
- Evolve the quantum state for a brief period of time with a Hamiltonian.

The first type of update is inconsequential. It involves contracting the matrix
that represents the local transformation with the tensor in the MPS that is
associated with the particular quantum subsystem.

However, the other operations are more complex and involve creating new tensors
that approximate in some way the outcome of the operation. We will focus now
on the two-site updates, such as the entangling gate mentioned in the second
item, because this is an operation that can be implemented accurately when an
MPS is in :doc:`canonical form <./tensor_canonical>`. Furthermore, this
transformation is the basis for more complex :doc:`evolution algorithms <tebd_evolution>`.

Let us consider a quantum operation :math:`U` acting on two quantum subsystems
that are "neighboring sites" in an MPS state representation. Say the first
subsystem is labeled :math:`i_n` and the second one is :math:`i_{n+1}`, in

.. math::
    |\psi\rangle = \cdots A_{\alpha_ni_n\alpha_{n+1}} B_{\alpha_{n+1}i_{n+1}\alpha_{n+2}}\cdots
    |\ldots i_n i_{n+1}\ldots\rangle

The quantum transformation can be represented as a four-legged tensor:

.. math::
    U = U_{j_{n}j_{n+1}i_{n}i_{n+1}}|j_nj_{n+1}\rangle\langle{i_ni_{n+1}}|


When we contract this operator with the MPS above, the two affected sites are
now associated to a combined tensor

.. math::
    U|\psi\rangle = \cdots C_{\alpha_nj_nj_{n+1}\alpha_{n+2}}\cdots
    |\ldots j_n j_{n+1}\ldots\rangle

where the larger, four-legged tensor is

.. math::
    C_{\alpha_nj_nj_{n+1}\alpha_{n+2}} =
     U_{j_{n}j_{n+1}i_{n}i_{n+1}} A_{\alpha_ni_n\alpha_{n+1}}
      B_{\alpha_{n+1}i_{n+1}\alpha_{n+2}}

and we assume the Einstein convention of summing over repeated indices,
:math:`i_n, i_{n+1}, \alpha_{n+1}`.

Our job is now to split the four-legged tensor :math:`C` into two new tensors,
:math:`\tilde{A}` and :math:`\tilde{B}`.

.. math::
    C_{\alpha_nj_nj_{n+1}\alpha_{n+2}} =
     \tilde{A}_{\alpha_nj_n\beta{n+1}}
     \tilde{B}_{\beta{n+1}j_{n+1}\alpha_{n+2}}

The tool for this, as usual, is the Schmidt decomposition. However, note that
now we are creating a new index :math:`\beta_{n+2}` which can be significantly
larger---it may be as large as the product of the dimensions of :math:`\alpha_{n}`
and :math:`j_n`, which represents an exponential growth!

It is therefore critical to implement this decomposition using a proper
truncation strategy, that drops the Schmidt values that have little or no
significance, and which keeps track of the truncation error caused by this
update algorithm.

In SeeMPS, two functions are responsible for this update. These are the following
two methods of the :class:`CanonicalMPS` class:

.. autosummary::

    ~seemps.state.CanonicalMPS.update_2site_right
    ~seemps.state.CanonicalMPS.update_2site_left

Both functions take as input the combined tensor :math:`C` and the site
`n` into which its decomposition is to be inserted. Both functions are also
destructive, in the sense that they change the state on which they operate.

The difference between these two methods is that they differ in which tensor,
:math:`A` or :math:`B` they associate to the :math:`U` and :math:`V` isometries.
If the state is in canonical form with respect to site `n` and you apply
`update_2site_right`, the updated state will be in canonical form with
respect to site `n+1`. Conversely, if the original state was in canonical
form with respect to site `n+1`, `update_2site_left` will leave its output
in canonical form with respect to `n`.

The following excerpt from :func:`~seemps.evolution.PairwiseUnitaries.apply_inplace`
shows how a list of two-site unitaries `U` is sequentially applied from "left"
to "right" on a quantum state in canonical form. The argument to the
`update_2site_right` function is an efficient contraction of the four-legged
tensor `U[i]` associated to the i-th unitary, and the corresponding tensors
of sites `j` and `j+1`. The result is an in-place update of the MPS `state`.

.. code-block:: python

    ...
    if center > 1:
        state.recenter(1)
    for j in range(L - 1):
        ## C = np.einsum("ijk,klm,nrjl -> inrm", state[j], state[j + 1], U[j])
        state.update_2site_right(
            _contract_nrjl_ijk_klm(U[j], state[j], state[j + 1]), j, strategy
        )
    ...