from typing import Any
from seemps.state import MPS, DEFAULT_STRATEGY, Strategy
from seemps.operators import MPO
from .problem import RKTypeEvolutionTestcase
from seemps.evolution import ODECallback, TimeSpan, runge_kutta


class TestRungeKutta(RKTypeEvolutionTestcase):
    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        return runge_kutta(
            L,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
        )
