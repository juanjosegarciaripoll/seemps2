.. currentmodule:: seemps.analysis.comptree


.. _alg_computation_tree:

******************************
Computation-tree constructions
******************************

Computation-tree constructions encode multivariate functions in MPS form by explicitly representing their algebraic evaluation as a directed tree (see Ref. https://arxiv.org/abs/2206.03832). Rather than approximating the function globally, these methods follow the structure of its computation, combining intermediate results locally and compressing them at each node.

In SeeMPS, computation trees are built by composing elementary nodes that apply binary functions to an incoming value and a local grid variable. Two main variants are supported:

1. Chain-like trees, represented by :class:`~ChainTree`.

2. Branching trees, represented by :class:`~BinaryTree`. 

Internal nodes are represented by :class:`~BranchNode`, while terminal nodes are given by :class:`~ChainRoot` for chain-like structures and :class:`~BinaryRoot` for branching structures. 

These structures are converted into MPS representations using :func:`~mps_chain_tree` and :func:`~mps_binary_tree`, respectively. At each node, intermediate images are optionally compressed, yielding highly sparse MPS cores that reflect the hierarchical structure of the computation. Computation-tree constructions are particularly effective for procedurally defined functions and functions with sharp features, where polynomial expansions and tensor cross interpolation may suffer from slow convergence or Gibbs-type artifacts.

.. autosummary::

    ~mps_chain_tree
    ~mps_binary_tree
    ~ChainTree
    ~BinaryTree
    ~ChainRoot
    ~BinaryRoot
    ~BranchNode
