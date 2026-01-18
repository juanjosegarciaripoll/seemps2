.. currentmodule:: seemps.analysis.sketching


.. _alg_sketching:

***********************
Sketching constructions
***********************

Sketching methods employ randomized projections, or *sketches*, to efficiently identify low-rank structure in large tensor representations. In the setting of function approximation, the "tensor-train recursive-sketching from samples" algorithm (TT-RSS, see Ref. https://arxiv.org/abs/2501.06300v1) combines tensor cross-interpolation ideas with sketching to construct MPS representations of functions from a fixed collection of samples. The resulting approach organizes the samples into tensor fibers, applies randomized projections to compress intermediate unfoldings, and reconstructs the tensor cores by solving a sequence of small least-squares problems.

The function :func:`~tt_rss` requires a :class:`~seemps.analysis.cross.BlackBoxLoadMPS` object, which encapsulates the target function together with its discretization using :class:`~seemps.analysis.mesh.Mesh` and MPS structure, thus relying on the same black-box interface as TCI (see :func:`~seemps.analysis.cross.cross_interpolation`). Additionally, it requires a collection of samples drawn from the function's domain. The algorithm returns an MPS approximation that fits the supplied samples, without requiring explicit access to the full tensor or additional function evaluations. The original implementation of TT-RSS, based on PyTorch, is available in the TensorKrowch library (see https://github.com/joserapa98/tensorkrowch).

Sketching-based constructions are particularly well suited for high-dimensional functions and probability densities, where polynomial expansions and tensor cross interpolation may fail to converge or become computationally prohibitive. By fitting the provided samples rather than reconstructing the target function globally, these methods control computational complexity at the expense of global approximation accuracy.

.. autosummary::

    ~tt_rss
