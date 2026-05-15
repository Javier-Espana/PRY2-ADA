from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import inf as Inf
from operator import or_ as BitwiseOr
from random import Random
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PainterSubset:
    Mask: int
    Cost: float
    Name: str = ""


@dataclass(frozen=True, slots=True)
class CoverResult:
    Feasible: bool
    TotalCost: float
    Selected: tuple[PainterSubset, ...]
    Algorithm: str


def SolveExactDp_Subsets(Painters: Sequence[PainterSubset], Sections: int) -> CoverResult:
    if Sections < 0:
        raise ValueError("Sections must be non-negative")

    Target = (1 << Sections) - 1
    InfCost = float("inf")
    Dp = [InfCost] * (1 << Sections)
    Parent: list[tuple[int, int] | None] = [None] * (1 << Sections)
    Dp[0] = 0.0

    for mask in range(1 << Sections):
        if Dp[mask] == InfCost:
            continue
        for idx, painter in enumerate(Painters):
            NewMask = mask | painter.Mask
            Candidate = Dp[mask] + painter.Cost
            if Candidate < Dp[NewMask]:
                Dp[NewMask] = Candidate
                Parent[NewMask] = (mask, idx)

    if Dp[Target] == InfCost:
        return CoverResult(False, Inf, tuple(), "DpSubsets")

    Chosen: list[PainterSubset] = []
    Cursor = Target
    while Cursor != 0:
        Entry = Parent[Cursor]
        if Entry is None:
            break
        PreviousMask, Index = Entry
        Chosen.append(Painters[Index])
        Cursor = PreviousMask

    Chosen.reverse()
    UnionMask = reduce(BitwiseOr, (Painter.Mask for Painter in Chosen), 0)
    Feasible = (UnionMask & Target) == Target
    TotalCost = sum(Painter.Cost for Painter in Chosen) if Feasible else Inf
    return CoverResult(Feasible, TotalCost, tuple(Chosen), "DpSubsets")


def SolveGreedyHeuristic_Subsets(Painters: Sequence[PainterSubset], Sections: int) -> CoverResult:
    if Sections < 0:
        raise ValueError("Sections must be non-negative")

    Target = (1 << Sections) - 1
    Chosen: list[PainterSubset] = []
    Covered = 0
    PaintersList = list(Painters)

    while Covered != Target:
        BestIndex = None
        BestScore = -1.0
        for Index, Painter in enumerate(PaintersList):
            NewBits = Painter.Mask & (~Covered)
            NewCount = NewBits.bit_count()
            if NewCount == 0:
                continue
            Score = NewCount / Painter.Cost
            if Score > BestScore:
                BestScore = Score
                BestIndex = Index

        if BestIndex is None:
            return CoverResult(False, Inf, tuple(), "GreedySubsets")

        Painter = PaintersList[BestIndex]
        Chosen.append(Painter)
        Covered |= Painter.Mask

    return CoverResult(True, sum(Painter.Cost for Painter in Chosen), tuple(Chosen), "GreedySubsets")


def GenerateFeasibleInstance_Subsets(Sections: int, PainterCount: int, Rng: Random) -> list[PainterSubset]:
    if Sections <= 0:
        raise ValueError("Sections must be positive")
    if PainterCount <= 0:
        raise ValueError("PainterCount must be positive")

    Painters: list[PainterSubset] = []
    for Section in range(Sections):
        Mask = 1 << Section
        Cost = float(Rng.randint(1, 3))
        Painters.append(PainterSubset(Mask=Mask, Cost=Cost, Name=f"b{Section}"))

    while len(Painters) < PainterCount:
        Mask = 0
        for Section in range(Sections):
            if Rng.random() < 0.5:
                Mask |= 1 << Section
        if Mask == 0:
            Mask = 1 << Rng.randint(0, Sections - 1)
        Cost = float(Rng.randint(1, 10))
        Painters.append(PainterSubset(Mask=Mask, Cost=Cost, Name=f"r{len(Painters)}"))

    UnionMask = 0
    for Painter in Painters:
        UnionMask |= Painter.Mask
    if UnionMask != (1 << Sections) - 1:
        Missing = ((1 << Sections) - 1) & ~UnionMask
        Section = 0
        while Missing:
            Bit = Missing & -Missing
            Missing &= Missing - 1
            Painters.append(PainterSubset(Mask=Bit, Cost=1.0, Name=f"fix{Section}"))
            Section += 1

    return Painters
