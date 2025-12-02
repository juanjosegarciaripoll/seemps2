from seemps.evolution.euler import euler, euler2, implicit_euler
from typing import Any
from seemps.state import MPS, DEFAULT_STRATEGY, Strategy
from seemps.operators import MPO
from .problem import RKTypeEvolutionTestcase
from seemps.evolution import ODECallback, TimeSpan


class TestEuler(RKTypeEvolutionTestcase):
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
        return euler(
            H,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
            itime=itime,
        )

    def accummulated_phase(self, E, dt, steps):
        return (1 - 1j * E * dt) ** steps


class TestEuler2(RKTypeEvolutionTestcase):
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
        return euler2(
            H,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
            itime=itime,
        )


class TestImplicitEuler(RKTypeEvolutionTestcase):
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
        return implicit_euler(
            H,
            time,
            state,
            steps=steps,
            strategy=strategy,
            callback=callback,
            itime=itime,
        )
