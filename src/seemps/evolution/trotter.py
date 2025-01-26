from __future__ import annotations
from abc import abstractmethod, ABC
from ..typing import Unitary
import scipy.linalg  # type: ignore
from ..hamiltonians import NNHamiltonian  # type: ignore
from ..state import Strategy, DEFAULT_STRATEGY, MPS, CanonicalMPS
from ..state._contractions import _contract_nrjl_ijk_klm


class PairwiseUnitaries:
    """Chain of unitaries acting on consecutive pairs of quantum subsystems.

    This class contains a set of operators :math:`U_{i,i+1} = exp(-i dt h_{i,i+1})`
    generated by the nearest-neighbor interactions :math:`h_{i,i+1}` of a 1D
    Hamiltonian.

    Parameters
    ----------
    H : NNHamiltonian
        The Hamiltonian with nearest-neighbor interactions generating the
        unitary transformations.
    dt : float
        Length of the time step.
    strategy : Strategy
        Truncation strategy for the application of the unitaries.
    """

    U: list[Unitary]

    def __init__(self, H: NNHamiltonian, dt: float, strategy: Strategy):
        self.U = [
            scipy.linalg.expm((-1j * dt) * H.interaction_term(k))
            for k in range(H.size - 1)
        ]
        self.strategy = strategy

    def apply(self, state: MPS) -> CanonicalMPS:
        """Apply the chain of unitaries onto a quantum state in MPS form.

        Parameters
        ----------
        state : MPS
            State to transform

        Returns
        -------
        CanonicalMPS
            A fresh new state with the new tensors.
        """
        return self.apply_inplace(
            state.copy() if isinstance(state, CanonicalMPS) else state
        )

    def apply_inplace(self, state: MPS) -> CanonicalMPS:
        """Apply the chain of unitaries onto a quantum state in MPS form,
        destructively modifying it.

        Parameters
        ----------
        state : MPS
            State to transform

        Returns
        -------
        CanonicalMPS
            The same `state` object.
        """
        strategy = self.strategy
        if not isinstance(state, CanonicalMPS):
            state = CanonicalMPS(state, center=0, strategy=strategy)
        L = state.size
        U = self.U
        center = state.center
        if center < L // 2:
            if center > 1:
                state.recenter(1)
            for j in range(L - 1):
                # AA = np.einsum("ijk,klm,nrjl -> inrm", state[j], state[j + 1], U[j])
                state.update_2site_right(
                    _contract_nrjl_ijk_klm(U[j], state[j], state[j + 1]), j, strategy
                )
        else:
            if center < L - 2:
                state.recenter(L - 2)
            for j in range(L - 2, -1, -1):
                # AA = np.einsum("ijk,klm,nrjl -> inrm", state[j], state[j + 1], U[j])
                state.update_2site_left(
                    _contract_nrjl_ijk_klm(U[j], state[j], state[j + 1]), j, strategy
                )
        return state


class Trotter(ABC):
    """Abstract class representing a Trotter TEBD algorithm."""

    @abstractmethod
    def apply(self, state: MPS) -> CanonicalMPS:
        """Apply this unitary onto an MPS `state`.

        This abstract method must be redefined by any children class.

        Parameters
        ----------
        state : MPS
            The state to be evolved.

        Returns
        -------
        CanonicalMPS
            A fresh new MPS wih the state evolved by one time step.
        """
        raise Exception("Called abstract method Trotter.apply")

    @abstractmethod
    def apply_inplace(self, state: MPS) -> CanonicalMPS:
        """Apply this unitary onto an MPS `state`, modifying it.

        This abstract method must be redefined by any children class.

        Parameters
        ----------
        state : MPS
            The state to be evolved.

        Returns
        -------
        CanonicalMPS
            The same `state` object, whenever possible.
        """
        raise Exception("Called abstract method Trotter.apply")

    def __matmul__(self, state: MPS) -> CanonicalMPS:
        return self.apply(state)


class Trotter2ndOrder(Trotter):
    """Class implementing a 2nd order Trotter algorithm.

    This class implements a Trotter algorithm in TEBD form for a nearest-neighbor
    interaction Hamiltonain :math:`\\sum_i h_{i,i+1}`. This second-order
    formula is a variation of the two-sweep algorithm

    .. math::
        \\Prod_{j=1}^{N} \\exp(-\\frac{i}{2} h_{j,j+1} dt)
        \\Prod_{j=N}^{1} \\exp(-\\frac{i}{2} h_{j,j+1} dt)

    Parameters
    ----------
    H : NNHamiltonian
        The Hamiltonian with nearest-neighbor interactions generating the
        unitary transformations.
    dt : float
        Length of the time step.
    strategy : Strategy
        Truncation strategy for the application of the unitaries."""

    U: PairwiseUnitaries
    strategy: Strategy

    def __init__(
        self, H: NNHamiltonian, dt: float, strategy: Strategy = DEFAULT_STRATEGY
    ):
        self.U = PairwiseUnitaries(H, 0.5 * dt, strategy)

    def apply(self, state: MPS) -> CanonicalMPS:
        """Apply a Trotter 2nd order unitary approximation onto an MPS `state`.

        Parameters
        ----------
        state : MPS
            The state to be evolved.

        Returns
        -------
        CanonicalMPS
            A fresh new MPS wih the state evolved by one time step.
        """
        state = self.U.apply(state)
        return self.U.apply_inplace(state)

    def apply_inplace(self, state: MPS) -> CanonicalMPS:
        """Apply a Trotter 2nd order unitary approximation onto an MPS `state`.

        Parameters
        ----------
        state : MPS
            The state to be evolved.

        Returns
        -------
        CanonicalMPS
            The same `state` object modified by the unitary, if it was a
            :class:`CanonicalMPS` Otherwise a fresh new state evolved.
        """
        state = self.U.apply_inplace(state)
        return self.U.apply_inplace(state)


class Trotter3rdOrder(Trotter):
    """Class implementing a 3rd order Trotter algorithm.

    This class implements a Trotter algorithm in TEBD form for a nearest-neighbor
    interaction Hamiltonain :math:`\\sum_i h_{i,i+1}`. This third-order
    formula is a variation of the three-sweep algorithm

    .. math::
        \\Prod_{j=1}^{N} \\exp(-\\frac{i}{4} h_{j,j+1} dt)
        \\Prod_{j=N}^{1} \\exp(-\\frac{i}{2} h_{j,j+1} dt)
        \\Prod_{j=1}^{N} \\exp(-\\frac{i}{4} h_{j,j+1} dt)

    Parameters
    ----------
    H : NNHamiltonian
        The Hamiltonian with nearest-neighbor interactions generating the
        unitary transformations.
    dt : float
        Length of the time step.
    strategy : Strategy
        Truncation strategy for the application of the unitaries."""

    U: PairwiseUnitaries
    Umid: PairwiseUnitaries

    def __init__(
        self,
        H: NNHamiltonian,
        dt: float,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.Umid = PairwiseUnitaries(H, 0.5 * dt, strategy)
        self.U = PairwiseUnitaries(H, 0.25 * dt, strategy)

    def apply(self, state: MPS) -> CanonicalMPS:
        """Apply a Trotter 2nd order unitary approximation onto an MPS `state`.

        Parameters
        ----------
        state : MPS
            The state to be evolved.

        Returns
        -------
        CanonicalMPS
            A fresh new MPS wih the state evolved by one time step.
        """
        state = self.U.apply(state)
        state = self.Umid.apply_inplace(state)
        return self.U.apply_inplace(state)

    def apply_inplace(self, state: MPS) -> CanonicalMPS:
        """Apply a Trotter 3rd order unitary approximation onto an MPS `state`.

        Parameters
        ----------
        state : MPS
            The state to be evolved.

        Returns
        -------
        CanonicalMPS
            The same `state` object modified by the unitary, if it was a
            :class:`CanonicalMPS` Otherwise a fresh new state evolved.
        """
        state = self.U.apply_inplace(state)
        state = self.Umid.apply_inplace(state)
        return self.U.apply_inplace(state)
