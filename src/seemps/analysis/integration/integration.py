from __future__ import annotations
import numpy as np

from ...state import MPS, scprod
from ...typing import Matrix, MPSOrder
from ..cross import cross_interpolation, CrossStrategyMaxvol, BlackBoxLoadMPS
from ..factories import mps_tensor_product
from ..mesh import Interval, RegularInterval, ChebyshevInterval, ArrayInterval, Mesh
from .mps_quadratures import (
    mps_trapezoidal,
    mps_simpson38,
    mps_fifth_order,
    mps_clenshaw_curtis,
    mps_fejer,
)
from .vector_quadratures import (
    vector_best_newton_cotes,
    vector_clenshaw_curtis,
    vector_fejer,
)


def integrate_mps(
    mps: MPS, domain: Interval | Mesh, mps_order: MPSOrder = "A"
) -> complex:
    """Compute the integral of a multivariate function represented as a MPS.

    The function is integrated over the discretization domain specified by an `Interval`
    or a `Mesh`. The appropriate univariate quadrature rule is selected automatically
    from the interval type:

        - `RegularInterval` → high-order Newton–Cotes rules.

        -  `ChebyshevInterval` → Clenshaw–Curtis or Fejér rules.

    The integral is then evaluated by contracting the MPS with the tensor-product
    quadrature weights, respecting the qubit ordering specified by `mps_order`.

    Parameters
    ----------
    mps : MPS
        MPS representation of the multivariate function to be integrated.
    domain : Interval | Mesh
        The discretization domain of the function. A `Mesh` is interpreted as a list
        of univariate intervals, each contributing its own quadrature rule.
    mps_order : MPSOrder, default='A'
        Ordering convention for the qubits: 'A' (serial) or 'B' (interleaved).

    Returns
    -------
    complex
        The value of the integral over the specified domain.

    Notes
    -----
    - All variables are assumed to use base-2 quantization on either a `RegularInterval`
      or a `ChebyshevInterval`, in serial or interleaved form.

    - More general quadrature operators can be built manually by forming the tensor
      product of univariate rules and contracting with `mps_tensor_product` followed by
      `scprod`.

    - Quadrature meshes can also be constructed automatically using cross-interpolation
      using :func:`quadrature_mesh_to_mps` in arbitrary tensor arrangements.

    Examples
    --------
    Integrate a bivariate function using Clenshaw–Curtis quadrature:

    .. code-block:: python

        mps_function_2d = ...  # MPS representation
        interval = ChebyshevInterval(-1, 1, 2**10, endpoints=True)
        mesh = Mesh([interval, interval])
        integral = integrate_mps(mps_function_2d, mesh)
    """
    mesh = domain if isinstance(domain, Mesh) else Mesh([domain])
    quads = []
    for interval in mesh.intervals:
        a, b, N = interval.start, interval.stop, interval.size
        n = int(np.log2(N))
        if isinstance(interval, RegularInterval):
            if n % 4 == 0:
                quads.append(mps_fifth_order(a, b, n))
            elif n % 2 == 0:
                quads.append(mps_simpson38(a, b, n))
            else:
                quads.append(mps_trapezoidal(a, b, n))
        elif isinstance(interval, ChebyshevInterval):
            if interval.endpoints:
                quads.append(mps_clenshaw_curtis(a, b, n))
            else:
                quads.append(mps_fejer(a, b, n))
        else:
            raise ValueError("Invalid interval in mesh")
    mps_quad = quads[0] if len(quads) == 1 else mps_tensor_product(quads, mps_order)
    return scprod(mps, mps_quad)


def mesh_to_quadrature_mesh(mesh: Mesh) -> Mesh:
    """
    Constructs a mesh whose entries are quadrature vectors derived from the best
    available quadrature rule for each `Interval` in the input mesh.

    Can be used to automatically construct a quadrature MPS using cross-interpolation
    with the :func:`quadrature_mesh_to_mps` routine in arbitrary tensor arrangements.
    """
    quads: list[Interval] = []
    for interval in mesh.intervals:
        start, stop, size = interval.start, interval.stop, interval.size

        if isinstance(interval, RegularInterval):
            quad = vector_best_newton_cotes(start, stop, size)
        elif isinstance(interval, ChebyshevInterval):
            if interval.endpoints:
                quad = vector_clenshaw_curtis(start, stop, size)
            else:
                quad = vector_fejer(start, stop, size)
        else:
            raise ValueError("Invalid Interval")
        quads.append(ArrayInterval(quad))

    return Mesh(quads)


def quadrature_mesh_to_mps(
    quadrature_mesh: Mesh,
    map_matrix: Matrix | None = None,
    physical_dimensions: list | None = None,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
    **kwargs,
) -> MPS:
    """
    Constructs the MPS representation of a multidimensional quadrature mesh using TCI.

    The input mesh consists of univariate quadrature vectors (or arbitrary 1D arrays)
    defining the weights along each dimension. These vectors are combined into a full
    multidimensional quadrature operator and approximated in MPS form using tensor
    cross-interpolation with the specified strategy.
    """
    black_box = BlackBoxLoadMPS(
        lambda q: np.prod(q, axis=0),
        quadrature_mesh,
        map_matrix,
        physical_dimensions,
    )
    return cross_interpolation(black_box, cross_strategy, **kwargs).mps
