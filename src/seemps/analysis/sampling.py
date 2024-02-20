from typing import Union
import numpy as np
from seemps.operators import MPO
from seemps.state import MPS


def sample_mps(mps: MPS, mps_indices: np.ndarray) -> np.ndarray:
    """
    Returns the samples corresponding to an array of MPS indices.

    Parameters
    ----------
    mps : MPS
        The MPS to sample from.
    mps_indices : np.ndarray
        An array of indices to be sampled on the MPS.

    Returns
    -------
    samples : np.ndarray
        The array of samples corresponding to the provided indices."""
    # TODO: Think about if this is redundant and whether the state/sampling.py module can be used instead.
    if mps_indices.ndim == 1:
        mps_indices = mps_indices[np.newaxis, :]
    reduced_mps = mps[0][0, mps_indices[:, 0], :]
    for i in range(1, len(mps)):
        # TODO: Replace einsum by something more efficient
        reduced_mps = np.einsum(
            "kq,qkr->kr", reduced_mps, mps[i][:, mps_indices[:, i], :]
        )
    samples = reduced_mps[:, 0]
    return samples


def random_mps_indices(
    mps: MPS,
    num_indices: int = 1000,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Generates random indices for sampling a MPS.

    Parameters
    ----------
    mps : MPS
        The matrix product state to sample from.
    num_indices : int, default 1000
        The number of random indices to generate.
    rng : np.random.Generator, default=`numpy.random.default_rng()`
        The random number generator to be used. If None, uses Numpy's
        default random number generator without any predefined seed.

    Returns
    -------
    indices : np.ndarray
        An array of random MPS indices."""
    return np.vstack([rng.choice(k, num_indices) for k in mps.physical_dimensions()]).T


def infinity_norm(tensor_network: Union[MPS, MPO], k_vals: int = 100) -> float:
    """
    Finds the infinity norm of a given tensor network.
    For a MPS, it corresponds to its largest element in absolute value.
    For a MPO, it corresponds to its operator norm (how much it 'lengthens' MPS).

    Parameters
    ----------
    tensor_network : Union[MPS, MPO]
        The given MPS or MPO to compute the norm.
    k_vals : int, default 100
        The number of top values to consider for finding the largest value.

    Returns
    -------
    norm : float
        The infinity norm of the tensor network.
    """
    N = len(tensor_network)
    if isinstance(tensor_network, MPO):
        mps: MPS = tensor_network @ MPS([np.ones((1, 2, 1))] * N)  # type: ignore
    elif isinstance(tensor_network, MPS):
        mps = tensor_network

    # TODO: Clean up
    scale_factor = 1 / N
    _, s, r = mps[0].shape
    I = np.arange(s).reshape(-1, 1)  # Physical indices of mps[0]
    reduced_site = mps[0].reshape(s, r) * 2**scale_factor
    for site in mps[1:]:
        _, s, r = site.shape
        # TODO: Replace einsum with a more efficient method
        reduced_site = np.einsum("kr,riq->kiq", reduced_site, site).reshape(-1, r)
        I_l = np.kron(I, np.ones((s, 1), dtype=int))
        I_r = np.kron(np.ones((I.shape[0], 1), dtype=int), np.arange(s).reshape(-1, 1))
        I = np.hstack((I_l, I_r))
        norms = np.sum((reduced_site / np.max(np.abs(reduced_site))) ** 2, axis=1)
        candidates = np.argsort(norms)[: -(k_vals + 1) : -1]
        I = I[candidates, :]
        reduced_site = reduced_site[candidates, :] * 2**scale_factor
    largest_idx = I[0]
    largest_value = sample_mps(mps, largest_idx)[0]
    norm = abs(largest_value)
    return norm
