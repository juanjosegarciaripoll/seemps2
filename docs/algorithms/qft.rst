.. _alg_qft:

**************************
Quantum Fourier Transform
**************************

The Quantum Fourier Transform (QFT) is the MPS equivalent of the Fast Fourier
Transform (FFT). It transforms vectors in a space of dimension :math:`2^n` according to:

.. math::
    \mathbf{e}_{i} \xrightarrow{\mathcal{F}} \frac{1}{\sqrt{2^n}}
    \sum_{j=0}^{2^n-1} e^{-i 2 \pi ij / 2^n} \mathbf{e}_j

In SeeMPS, the Fourier transform is implemented as a sequence of unitary
transformations :math:`\mathcal{F} = F_n F_{n-1} \cdots F_1` that mimic layers
of a quantum Fourier transform circuit acting on :math:`n` qubits, up to the
final qubit swap.

Implementation
==============

The algorithm assumes that the Fourier transform acts on an MPS composed of
:math:`n` two-dimensional objects (qubits). The total algorithm is encoded as
an :class:`~seemps.operators.MPOList` with :math:`n` MPOs implementing Hadamard
gates and conditional rotations, with exact tensors of small bond dimension.

Specifically, for the :math:`i`-th layer:

- Sites :math:`n < i` have identity tensors
- Site :math:`i` has a Hadamard gate with control output
- Sites :math:`j > i` have conditional rotation gates :math:`\exp(i 2\pi / 2^{j-i})`

Bond dimension preservation
===========================

An important property of the QFT in the MPS formalism is that it does not
significantly increase the bond dimension of the state. This was first reported
for discrete encodings of bandwidth-limited functions and later confirmed in
more general scenarios. This property enables efficient use of Fourier-based
techniques for:

- Spectral differentiation (see :doc:`../seemps_analysis_differentiation`)
- Fourier interpolation (see :doc:`../seemps_analysis_interpolation`)
- Exponential speedups in certain applications

Qubit reversal
==============

The QFT circuit produces output qubits in reversed order compared to the
standard FFT convention. The function :func:`~seemps.qft.qft_flip` implements
the qubit reversal that makes ``qft_flip(qft(f))`` equivalent to the FFT of
the vector version of ``f``. However, this operation can be costly in the MPS
representation, so it is often avoided when possible.

Negative frequencies in the output are placed in the upper part of the quantum
register, following the two's complement notation.

Multidimensional transforms
===========================

For multidimensional functions encoded as MPS, SeeMPS provides partial transforms
that act on subsets of qubits. The functions :func:`~seemps.qft.qft_nd_mpo` and
:func:`~seemps.qft.iqft_nd_mpo` create transforms for specified qubit indices,
enabling efficient multidimensional Fourier analysis.

.. autosummary::

   ~seemps.qft.qft
   ~seemps.qft.iqft
   ~seemps.qft.qft_mpo
   ~seemps.qft.iqft_mpo
   ~seemps.qft.qft_nd_mpo
   ~seemps.qft.iqft_nd_mpo
   ~seemps.qft.qft_flip

See also
========

- :doc:`../seemps_analysis_differentiation` - Fourier-based differentiation methods
- :doc:`../seemps_analysis_interpolation` - Fourier interpolation techniques
