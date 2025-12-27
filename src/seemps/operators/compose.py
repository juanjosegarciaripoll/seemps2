from __future__ import annotations

from ..state import mps_tensor_product, Strategy
from ..typing import MPSOrder
from .mpo import MPO
from .simplify_mpo import mps_as_mpo, mpo_as_mps


def mpo_tensor_product(
    mpo_list: list[MPO],
    mpo_order: MPSOrder = "A",
    strategy: Strategy | None = None,
    simplify_steps: bool = False,
) -> MPO:
    return mps_as_mpo(
        mps_tensor_product(
            [mpo_as_mps(O) for O in mpo_list], mpo_order, strategy, simplify_steps
        )
    )
