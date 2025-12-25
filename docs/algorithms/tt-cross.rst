.. currentmodule:: seemps.analysis.cross

.. _alg_ttcross:

*******************************************
Tensor-train cross-interpolation (TT-Cross)
*******************************************

Tensor-train cross-interpolation, known as TT-Cross or TCI, is a method that computes the tensor-train representation of a black-box function by sampling some of its elements along some patterns known as crosses. As it does not act on the explicit tensor representation, it provides an exponential advantage over the standard Schmidt decomposition and evades the *curse of dimensionality*.

There are several variants available for TT-Cross. Each shows different advantages and disadvantages in terms of computational cost and accuracy for different initial conditions. This library implements three:

1. :func:`~cross_dmrg`: Based on two-site optimizations combining the skeleton decomposition and the Schmidt decomposition. It is efficient for structures of low physical dimension, such as binary MPS, as it can increase the bond dimension by several units for each sweep. Inefficient for structures of large physical dimension due to its computational complexity. Has an associated parameter dataclass given by :class:`~CrossStrategyDMRG`.

2. :func:`~cross_greedy`: Based on two-site optimizations performing greedy searches for maximum-volume pivots. Efficient for structures of large physical dimension, such as tensor-trains with dense modes, due to its advantageous computational complexity. Inefficient for structures of reduced physical dimension, as it increases the bond dimension by one each sweep. Presents the ``full search`` variant or the ``partial search`` variants. Has an associated parameter dataclass given by :class:`~CrossStrategyGreedy`.

3. :func:`~cross_maxvol`: Based on rank-adaptive one-site optimizations using the rectangular skeleton decomposition. Can be seen as a middle ground between the former two methods. Has an associated parameter dataclass given by :class:`~CrossStrategyMaxvol`.

Moreover, this method performs the decomposition of a given input black-box. This black-box can take several different forms and serve for different application domains. This library implements the class :class:`~black_box.BlackBox` and the following subclasses:

1. :class:`~BlackBoxLoadMPS`: Required to load functions with quantized degrees of freedom in MPS. Allows for both the *serial* and *interleaved* qubit orders.

2. :class:`~BlackBoxLoadMPO`: Required to load bivariate functions in MPO, by computing the equivalent MPS representation. This MPS is of square physical dimension (e.g. 4 for a MPO of dimenson 2) and can be unfolded in the end to the required MPO.

3. :class:`~BlackBoxComposeMPS`: Required to compose scalar functions on collections of MPS.

An example on how to use TCI for all these scenarios is shown in `TT-Cross.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/TT-Cross.ipynb>`_.

.. autosummary::

    ~cross_dmrg
    ~cross_greedy
    ~cross_maxvol
    ~BlackBoxLoadMPS
    ~BlackBoxLoadMPO
    ~BlackBoxComposeMPS
    ~CrossStrategyDMRG
    ~CrossStrategyGreedy
    ~CrossStrategyMaxvol
