import numpy as np
from copy import deepcopy
from typing import Callable, List, Optional

from .maxvol import maxvol
from .mesh import Mesh
from ..state import MPS, random_mps


class Cross:
    """Cross class.

    This implements a Cross object that loads a multivariate function
    given by a Function object into a Matrix Product State with $n_i$ sites
    for each dimension by means of the Tensor Train Cross Interpolation (TT-Cross)
    algorithm with a sweeping scheme.

    Parameters
    ----------
    func : Function
        A scalar multivariate function acting on m-dimensional vectors.
    mps0 : MPS
        The starting MPS on which to load the function. Its shape must match that of the function.
        The algorithm will start using a bond dimension equal to that of mps0, but will grow it
        after each sweep. It can be either on binary or 'TT form'.
    options : dictionary, optional
        A dictionary containing options for the algorithm. Every empty option will be assigned a
        pre-defined value.

    """

    def __init__(
        self, func: Callable, mesh: Mesh, mps0: Optional[MPS] = None, options: dict = {}
    ):
        self.func = func
        self.mesh = mesh

        self.stop_options = {
            "min_error": options.get("min_error", 1e-10),
            "max_calls": options.get("max_calls", 1e7),
            "max_sweeps": options.get("max_sweeps", 1e3),
            "max_rank": options.get("max_rank", 1e3),
        }
        self.maxvol_options = {
            "min_rank_change": options.get("min_rank_change", 0),
            "max_rank_change": options.get("max_rank_change", 1),
            "tau0": options.get("tau0", 1.05),
            "k0": options.get("k0", 100),
            "tau": options.get("tau", 1.1),
        }
        self.measure_options = {
            "measure_type": options.get("measure_type", "sampling"),
            "sampling_points": options.get("sampling_points", 1000),
            "verbose": options.get("verbose", True),
        }
        self.mps_options = {
            "structure": options.get("structure", "binary"),
            "ordering": options.get("ordering", "A"),
            "starting_rank": options.get("starting_rank", 5),
            "starting_distribution": options.get("starting_distribution", "random"),
        }

        if mps0 is None:
            qubits = self.mesh.qubits
            D = self.mps_options["starting_rank"]
            if self.mps_options["structure"] == "binary":
                n = sum(qubits)
                self.mps0 = random_mps([2] * n, D)
            elif self.mps_options["structure"] == "tt":
                points = [2**n for n in qubits]
                self.mps0 = random_mps(points, D)
            else:
                raise ValueError("Invalid option 'structure'")
        else:
            self.mps0 = mps0

        self.I_physical = [
            np.reshape(np.arange(k, dtype=int), (-1, 1))
            for k in self.mps0.physical_dimensions()
        ]

        self.tracker = {
            "error": [1],
            "calls": [0],
            "sweeps": [0],
            "rank_max": [max(self.mps0.bond_dimensions())],
        }

    # TODO: Clean and optimize
    def binary_to_decimal_map(self, binary_indices, qubits):
        """Maps an array of multi-indices in binary form to an array of
        multi-indices in decimal form."""

        def bitlist_to_int(bitlist):
            """
            Fast map between binary an decimal numbers:
            https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
            """
            out = 0
            for bit in bitlist:
                out = (out << 1) | bit
            return out

        m = len(qubits)
        decimal_indices = []
        ordering = self.mps_options["ordering"]
        for idx, n in enumerate(qubits):
            if ordering == "A":
                rng = np.arange(idx * n, (idx + 1) * n)
            elif ordering == "B":
                rng = np.arange(idx, m * n, m)
            else:
                raise ValueError("Invalid ordering")
            decimal_ndx = bitlist_to_int(binary_indices.T[rng])
            decimal_indices.append(decimal_ndx)

        decimal_indices = np.column_stack(decimal_indices)
        return decimal_indices

    def reorder_tensor(tensor, qubits):
        """Reorders an A-ordered tensor into a B-ordered tensor."""
        m = len(qubits)
        tensor = tensor.reshape([2] * sum(qubits))
        axes = [np.arange(idx, m * n, m) for idx, n in enumerate(qubits)]
        axes = [item for items in axes for item in items]
        tensor = np.transpose(tensor, axes=axes)
        return tensor

    def get_sampling_error(self, Y, sampling_indices, sampled_vector):
        """Measures the sampling error between the MPS tensors, Y, at certain
        sampling indices, and the vector of exact function samples.
        """
        Q = Y[0][0, sampling_indices[:, 0], :]
        for i in range(1, len(Y)):
            Q = np.einsum("kq,qkr->kr", Q, Y[i][:, sampling_indices[:, i], :])
        y_tt = Q[:, 0]
        error = np.linalg.norm(y_tt - sampled_vector) / np.linalg.norm(sampled_vector)
        return error

    def get_norm_error(self, Y, norm_prev):
        """Measures the norm error between the MPS tensors, Y, and the
        norm of a previous iteration given by norm_prev."""
        norm = MPS(Y).norm()
        error = abs(norm_prev - norm)
        return norm, error

    def presweep(self, Y: List[np.ndarray]):
        """Runs a presweep on the MPS tensors, Y.
        Initializes the multi-indices I_forward and I_backward.
        """
        n = len(Y)
        I_forward = [None for _ in range(n + 1)]
        I_backward = [None for _ in range(n + 1)]

        maxvol_options = deepcopy(self.maxvol_options)
        maxvol_options.update({"min_rank_change": 0, "max_rank_change": 0})

        # Forward pass
        R = np.ones((1, 1))
        for j in range(n):
            G = np.tensordot(R, Y[j], 1)
            Y[j], I_forward[j + 1], R = maxvol(
                G, self.I_physical[j], I_forward[j], maxvol_options, ltr=True
            )
        Y[n - 1] = np.tensordot(Y[n - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(n - 1, -1, -1):
            G = np.tensordot(Y[j], R, 1)
            Y[j], I_backward[j], R = maxvol(
                G, self.I_physical[j], I_backward[j + 1], maxvol_options, ltr=False
            )
        Y[0] = np.tensordot(R, Y[0], 1)

        return Y, I_forward, I_backward

    def sweep(
        self,
        Y: List[np.ndarray],
        I_forward: List[np.ndarray],
        I_backward: List[np.ndarray],
    ):
        """Core of the cross interpolation algorithm.
        Runs a forward-backward sweep on the MPS tensors.
        Updates the tensors, forward and backward multiindices iteratively by means of the maxvol algorithm.
        Updates the tracker dictionary with sweep information.
        """
        d = len(Y)

        # Forward pass
        R = np.ones((1, 1))
        for j in range(d):
            G = self.evaluate(self.I_physical[j], I_forward[j], I_backward[j + 1])
            Y[j], I_forward[j + 1], R = maxvol(
                G, self.I_physical[j], I_forward[j], self.maxvol_options, ltr=True
            )
        Y[d - 1] = np.tensordot(Y[d - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(d - 1, -1, -1):
            G = self.evaluate(self.I_physical[j], I_forward[j], I_backward[j + 1])
            Y[j], I_backward[j], R = maxvol(
                G, self.I_physical[j], I_backward[j + 1], self.maxvol_options, ltr=False
            )
        Y[0] = np.tensordot(R, Y[0], 1)

        # Update tracker
        sweep = self.tracker["sweeps"][-1]
        self.tracker["sweeps"].append(sweep + 1)
        self.tracker["rank_max"].append(max(MPS(Y).bond_dimensions()))

        return Y, I_forward, I_backward

    def sample(self, indices: np.ndarray) -> np.ndarray:
        """Samples the multivariate function on an array of MPS multi-indices and
        returns a vector of function samples.
        If the MPS is binary, this first converts the indices to a decimal form."""
        if self.mps_options["structure"] == "binary":
            indices = self.binary_to_decimal_map(indices, self.mesh.qubits)
        sampled_vector = np.array([self.func(self.mesh[ndx]) for ndx in indices])
        return sampled_vector

    def evaluate(self, i_physical, i_forward, i_backward) -> np.ndarray:
        """Evaluates the multivariate function on a fiber of the MPS, given
        by the subtensor generated by all the physical indices on the site i
        together with the forward and backward multi-indices of the site i."""
        r1 = i_forward.shape[0] if i_forward is not None else 1
        s = i_physical.shape[0]
        r2 = i_backward.shape[0] if i_backward is not None else 1
        matrix = np.kron(np.kron(_ones(r2), i_physical), _ones(r1))

        if i_forward is not None:
            matrix_forward = np.kron(_ones(s * r2), i_forward)
            matrix = np.hstack((matrix_forward, matrix))
        if i_backward is not None:
            matrix_backward = np.kron(i_backward, _ones(r1 * s))
            matrix = np.hstack((matrix, matrix_backward))

        sweep = self.tracker["sweeps"][-1]
        if len(self.tracker["calls"]) < (sweep + 1):
            self.tracker["calls"].append(0)
        self.tracker["calls"][sweep] += len(matrix)
        tensor = self.sample(matrix)
        tensor = np.reshape(tensor, (r1, s, r2), order="F")
        return tensor

    def measure(self, Y: List[np.ndarray]) -> None:
        """Measures the MPS following a given method (sampling, norm, integral, etc.).
        Evaluates an error, updates the tracker dictionary and optionally prints the progress.
        """
        if self.measure_options["measure_type"] == "sampling":
            if self.tracker["sweeps"][-1] == 0:
                shape = self.mps0.physical_dimensions()
                sampling_points = self.measure_options["sampling_points"]
                self.sampling_indices = np.vstack(
                    [np.random.choice(k, sampling_points) for k in shape]
                ).T
                self.sampled_vector = self.sample(self.sampling_indices)
            error = self.get_sampling_error(
                Y, self.sampling_indices, self.sampled_vector
            )
            self.tracker["error"].append(error)
            title = "Sampling error"
        elif self.measure_options["measure_type"] == "norm":
            if self.tracker["sweeps"][-1] == 0:
                self.norm = 0
            self.norm, error = self.get_norm_error(Y, self.norm)
            self.tracker["error"].append(error)
            title = "Norm error"
        else:
            raise ValueError("Invalid measurement type")

        if self.measure_options["verbose"]:
            print(
                f'Sweep {self.tracker["sweeps"][-1]:<3} | '
                + f"Max χ {self.tracker['rank_max'][-1]:>3} | "
                + f"{title} {error:.2E} | "
                + f'Function calls {self.tracker["calls"][-1]:>8}'
            )

    def is_converged(self) -> None:
        """Checks if the convergence criteria is met in order to halt the algorithm."""
        return (
            self.tracker["error"][-1] < self.stop_options["min_error"]
            or self.tracker["calls"][-1] >= self.stop_options["max_calls"]
            or self.tracker["sweeps"][-1] >= self.stop_options["max_sweeps"]
            or self.tracker["rank_max"][-1] >= self.stop_options["max_rank"]
        )

    def run(self) -> MPS:
        """Runs the cross interpolation algorithm."""
        mps = deepcopy(self.mps0)
        Y = mps._data
        Y, I_forward, I_backward = self.presweep(Y)
        self.measure(Y)
        while not self.is_converged():
            Y, I_forward, I_backward = self.sweep(Y, I_forward, I_backward)
            self.measure(Y)
        print("Converged!")
        return mps, self.tracker


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)
