from seemps.evolution.euler import euler, euler2, implicit_euler
from typing import Any
from seemps.state import MPS, DEFAULT_STRATEGY, Strategy
from seemps.operators import MPO
from .problem import RKTypeEvolutionTestcase
from seemps.evolution import ODECallback, TimeSpan


class TestEuler(RKTypeEvolutionTestcase):
    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        return euler(
            L,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
        )

    def accumulated_amplification(self, E, dt, steps):
        return (1 + E * dt) ** steps


class TestEuler2(RKTypeEvolutionTestcase):
    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        return euler2(
            L,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
        )

    def accumulated_amplification(self, E, dt, steps):
        return (1 + E * dt + 0.5 * (E * dt) ** 2) ** steps


class TestImplicitEuler(RKTypeEvolutionTestcase):
    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        return implicit_euler(
            L,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
        )

    def accumulated_amplification(self, E, dt, steps):
        return ((1 + 0.5 * E * dt) / (1 - 0.5 * E * dt)) ** steps
