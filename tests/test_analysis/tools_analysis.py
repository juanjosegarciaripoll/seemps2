import numpy as np


def reorder_tensor(tensor, sites_per_dimension):
    """
    Reorders a given tensor between the MPS orderings 'A' and 'B' by transposing its axes.
    """
    dimensions = len(sites_per_dimension)
    shape_orig = tensor.shape
    tensor = tensor.reshape([2] * sum(sites_per_dimension))
    axes = [
        np.arange(idx, dimensions * n, dimensions)
        for idx, n in enumerate(sites_per_dimension)
    ]
    axes = [item for items in axes for item in items]
    tensor = np.transpose(tensor, axes=axes)
    return tensor.reshape(shape_orig)
