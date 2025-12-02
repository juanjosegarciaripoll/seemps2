from seemps.evolution.arnoldi import arnoldi
from typing import Any
from seemps.state import MPS, DEFAULT_STRATEGY, Strategy
from seemps.operators import MPO
from .problem import RKTypeEvolutionTestcase
from seemps.evolution import ODECallback, TimeSpan


class TestArnoldi(RKTypeEvolutionTestcase):
    def solve_Schroedinger(
        self,
        H: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
        itime: bool = False,
    ) -> MPS | list[Any]:
        return arnoldi(
            H,
            time,
            state,
            steps=steps,
            order=6,
            strategy=strategy,
            callback=callback,
            itime=itime,
        )
