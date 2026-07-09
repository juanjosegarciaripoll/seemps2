from typing import Any
from seemps.state import MPS, DEFAULT_STRATEGY, Strategy
from seemps.operators import MPO
from seemps.evolution.crank_nicolson import crank_nicolson
from .problem import RKTypeEvolutionTestcase
from seemps.evolution import ODECallback, TimeSpan


class TestCrankNicolson(RKTypeEvolutionTestcase):
    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        return crank_nicolson(
            L,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
        )

    def accumulated_amplification(self, E, dt, steps):
        return ((1 + 0.5 * E * dt) / (1 - 0.5 * E * dt)) ** steps
