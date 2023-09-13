import numpy as np
from .typing import *
from math import sqrt
from .tools import σx, σy, σz
import scipy.sparse as sp  # type: ignore
from abc import abstractmethod
from .mpo import MPO
from .state import schmidt, core, DEFAULT_STRATEGY, Strategy


class NNHamiltonian(object):
    """Abstract class representing a Hamiltonian for a 1D system with
    nearest-neighbor interactions.

    The Hamiltonian is assumed to have the structure

    .. math::
        H = \\sum_{i=0}^{N-2} h_{i,i+1}

    where each :math:`h_{i,i+1}` is a matrix acting on two quantum
    subsystems. Descendents from this class must implement both the
    :py:meth:`dimension` and :py:meth:`interaction_term` methods.
    """

    size: int
    """Number of quantum systems in this Hamiltonian."""

    constant: bool
    """True if the Hamiltonian does not depend on time."""

    def __init__(self, size: int):
        #
        # Create a nearest-neighbor interaction Hamiltonian
        # of a given size, initially empty.
        #
        self.size = size
        self.constant = False

    @abstractmethod
    def dimension(self, i: int) -> int:
        """Return the physical dimension of the `i`-th quantum system."""
        pass

    def interaction_term(self, i: int, t: float = 0.0) -> Operator:
        """Return the Operator acting on sites `i` and `i+1`.

        Parameters
        ----------
        i : int
            Index into the range `[0, self.size-1)`
        t : float
            Time at which this interaction matrix is computed

        Returns
        -------
        Operator
            Some type of matrix in tensor or sparse-matrix form.
        """
        return 0

    def tomatrix(self, t: float = 0.0) -> Operator:
        """Compute the sparse matrix for this Hamiltonian at time `t`.

        Parameters
        ----------
        t : float, default = 0.0
            Time at which the matrix is computed

        Returns
        -------
        Operator
            Matrix in either dense or sparse representation.
        """
        # dleft is the dimension of the Hilbert space of sites 0 to (i-1)
        # both included
        dleft = 1
        # H is the Hamiltonian of sites 0 to i, this site included.
        H = 0 * sp.eye(self.dimension(0))
        for i in range(self.size - 1):
            # We extend the existing Hamiltonian to cover site 'i+1'
            H = sp.kron(H, sp.eye(self.dimension(i + 1)))
            # We add now the interaction on the sites (i,i+1)
            H += sp.kron(sp.eye(dleft if dleft else 1), self.interaction_term(i, t))
            # We extend the dimension covered
            dleft *= self.dimension(i)

        return H

    def to_mpo(self, t: float = 0.0, strategy: Strategy = DEFAULT_STRATEGY) -> MPO:
        """Compute the matrix-product operator for this Hamiltonian at time `t`.

        Parameters
        ----------
        t : float, default = 0.0
            Time at which the Hamiltonian is evaluated.
        strategy : Strategy
            Truncation strategy for MPO tensors (defaults to DEFAULT_STRATEGY)

        Returns
        -------
        MPO
            Matrix-product operator.
        """
        tensors = [
            np.zeros((2, di, di, 2))
            for i in range(self.size)
            for di in [self.dimension(i)]
        ]
        for i in range(self.size - 1):
            Hij = np.asarray(self.interaction_term(i, t))
            di = self.dimension(i)
            dj = self.dimension(i + 1)
            Hij = (
                Hij.reshape(di, dj, di, dj)
                .transpose(0, 2, 1, 3)
                .reshape(di * di, dj * dj)
            )
            U, s, V = schmidt.svd(
                Hij,
                full_matrices=False,
                overwrite_a=True,
                check_finite=False,
                lapack_driver=schmidt.SVD_LAPACK_DRIVER,
            )
            s, _ = core.truncate_vector(s, strategy)
            ds = s.size
            s = np.sqrt(s)
            #
            # Extend the dimension of the tensor to include the
            # new interaction
            A = tensors[i]
            a, j, k, b = A.shape
            B = np.zeros((a, j, k, b + ds), dtype=U.dtype)
            B[:, :, :, :b] = A
            B[0, :, :, 2:] = (U[:, :ds] * s).reshape(di, di, ds)
            B[1, :, :, 1] = np.eye(j)
            B[0, :, :, 0] = np.eye(j)
            tensors[i] = B
            #
            # Similar operation for the next tensor
            A = tensors[i + 1]
            a, j, k, b = A.shape
            B = np.zeros((a + ds, j, k, b), dtype=V.dtype)
            B[:a, :, :, :] = A
            B[2:, :, :, 1] = (s.reshape(ds, 1) * V[:ds, :]).reshape(ds, dj, dj)
            B[1, :, :, 1] = np.eye(j)
            B[0, :, :, 0] = np.eye(j)
            tensors[i + 1] = B
        tensors[0] = tensors[0][[0], :, :, :]
        tensors[-1] = tensors[-1][:, :, :, [1]]
        return MPO(tensors)


class ConstantNNHamiltonian(NNHamiltonian):
    """Nearest-neighbor 1D Hamiltonian with constant terms.

    Parameters
    ----------
    size: int
        Number of quantum systems that this model is formed of.
    dimension: int | list[int]
        Either an integer denoting the dimension for all quantum subsystems,
        or a list of dimensions for each of the `size` objects.
    """

    dimensions: list[int]
    """List of dimensions of the quantum system."""

    interactions: list[Operator]
    """List of operators :math:`h_{i,i+1}`."""

    def __init__(self, size: int, dimension: Union[int, list[int]]):
        #
        # Create a nearest-neighbor interaction Hamiltonian with fixed
        # local terms and interactions.
        #
        #  - local_term: operators acting on each site (can be different for each site)
        #  - int_left, int_right: list of L and R operators (can be different for each site)
        #
        super(ConstantNNHamiltonian, self).__init__(size)
        self.constant = True
        if isinstance(dimension, list):
            self.dimensions = dimension
        else:
            self.dimensions = [dimension] * size
        self.interactions = [
            np.zeros((si * sj, si * sj))
            for si, sj in zip(self.dimensions[:-1], self.dimensions[1:])
        ]

    def add_local_term(self, site: int, operator: Operator) -> "ConstantNNHamiltonian":
        """Upgrade this Hamiltonian with a local term acting on `site`.

        Parameters
        ----------
        site : int
            The site on which this operator acts.
        operator : Operator
            The operator in dense or sparse form

        Returns
        -------
        ConstantNNHamiltonian
            This same object, modified to account for this extra term.
        """
        #
        # Set the local term acting on the given site
        #
        if site < 0 or site >= self.size:
            raise IndexError("Site {site} out of bounds in add_local_term()")
        if site == 0:
            self.add_interaction_term(site, operator, np.eye(self.dimensions[1]))
        elif site == self.size - 1:
            self.add_interaction_term(
                site - 1, np.eye(self.dimensions[site - 1]), operator
            )
        else:
            self.add_interaction_term(
                site - 1, np.eye(self.dimensions[site - 1]), 0.5 * operator
            )
            self.add_interaction_term(
                site, 0.5 * operator, np.eye(self.dimensions[site + 1])
            )
        return self

    def add_interaction_term(
        self, i: int, op1: Operator, op2: Optional[Operator] = None
    ) -> "ConstantNNHamiltonian":
        """Add an interaction term to this Hamiltonian, acting in 'site' and 'site+1'.
        If 'op2' is None, then 'op1' is interpreted as an operator acting on both
        sites in matrix form. If 'op1' and 'op2' are both provided, the operator
        is np.kron(op1, op2).

        Parameters
        ----------
        site : int
            First site of two (`site` and `site+1`) on which this interaction
            term acts.
        op1 : Operator
        op2 : Operator, optional
            (Default value = None)
            If `op2` is not supplied, then `op1` is the complete Hamiltonian
            :math:`h_{i,i+1}`. Otherwise, the Hamiltonian is the Kronecker
            product of `op1` and `op2`

        Returns
        -------
        ConstantNNHamiltonian
            This same object.
        """
        if i < 0 or i >= self.size - 1:
            raise IndexError("Site {site} out of bounds in add_interaction_term()")
        H12 = op1 if op2 is None else sp.kron(op1, op2)
        if (
            H12.ndim != 2
            or H12.shape[0] != H12.shape[1]
            or H12.shape[1] != self.dimension(i) * self.dimension(i + 1)
        ):
            raise Exception(f"Invalid operators supplied to add_interaction_term()")
        self.interactions[i] = self.interactions[i] + H12
        return self

    def dimension(self, i) -> int:
        return self.dimensions[i]

    def interaction_term(self, i: int, t: float = 0.0) -> Operator:
        """Return the same interaction term for sites `i` and `i+1`."""
        return self.interactions[i]


class ConstantTIHamiltonian(ConstantNNHamiltonian):
    """Translationally invariant Hamiltonian with constant nearest-neighbor
    interactions.

    Parameters
    ----------
    size: int
        Number of subsystems on which this Hamiltonian acts.
    interaction: Operator, optional
        Matrix for nearest-neighbor interactions
    local_term: Operator, optional
        Possible additional local term acting on each subsystem.
    """

    def __init__(
        self,
        size: int,
        interaction: Optional[Operator] = None,
        local_term: Optional[Operator] = None,
    ):
        if local_term is not None:
            dimension = len(local_term)
        elif interaction is not None:
            dimension = round(sqrt(interaction.shape[0]))
        else:
            raise Exception("Either interactions or local term must be supplied")

        super().__init__(size, dimension)
        for site in range(size - 1):
            if interaction is not None:
                self.add_interaction_term(site, interaction)
            if local_term is not None:
                self.add_local_term(site, local_term)


class HeisenbergHamiltonian(ConstantTIHamiltonian):
    """Nearest-neighbor Hamiltonian with constant Heisenberg interactions
    over 'size' S=1/2 spins.

    Parameters
    ----------
    size: int
        Number of spins on which this Hamiltonian operates.
    """

    def __init__(self, size: int):
        return super().__init__(
            size, 0.25 * (sp.kron(σx, σx) + sp.kron(σy, σy) + sp.kron(σz, σz)).real
        )
