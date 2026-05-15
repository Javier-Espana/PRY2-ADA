from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import inf as Inf
from operator import or_ as BitwiseOr
from random import Random
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Interval:
    Left: int
    Right: int
    Cost: float
    Name: str = ""

    @property
    def Length(self) -> int:
        return self.Right - self.Left + 1

    @property
    def Efficiency(self) -> float:
        return self.Length / self.Cost


@dataclass(frozen=True, slots=True)
class CoverResult:
    Feasible: bool
    TotalCost: float
    Intervals: tuple[Interval, ...]
    Algorithm: str


def NormalizeIntervals(Intervals: Iterable[Interval], FenceLength: int) -> list[Interval]:
    if FenceLength < 0:
        raise ValueError("FenceLength must be non-negative")

    Normalized: list[Interval] = []
    for IntervalItem in Intervals:
        if IntervalItem.Cost <= 0:
            raise ValueError("Interval costs must be positive")
        Left = max(0, IntervalItem.Left)
        Right = min(FenceLength, IntervalItem.Right)
        if Left > Right:
            continue
        Normalized.append(Interval(Left=Left, Right=Right, Cost=float(IntervalItem.Cost), Name=IntervalItem.Name))
    return Normalized


def CoverIsValid(Intervals: Sequence[Interval], FenceLength: int) -> bool:
    if FenceLength < 0:
        return False
    SortedIntervals = sorted(Intervals, key=lambda IntervalItem: (IntervalItem.Left, IntervalItem.Right, IntervalItem.Cost))
    NextUncovered = 0
    for IntervalItem in SortedIntervals:
        if IntervalItem.Right < NextUncovered:
            continue
        if IntervalItem.Left > NextUncovered:
            return False
        NextUncovered = IntervalItem.Right + 1
        if NextUncovered > FenceLength:
            return True
    return NextUncovered > FenceLength


def SolutionCost(Intervals: Sequence[Interval]) -> float:
    return sum(IntervalItem.Cost for IntervalItem in Intervals)


def SolveExactDp(Intervals: Sequence[Interval], FenceLength: int) -> CoverResult:
    Normalized = NormalizeIntervals(Intervals, FenceLength)

    Dp = [Inf] * (FenceLength + 1)
    Parent: list[int | None] = [None] * (FenceLength + 1)

    for CoveredUntil in range(FenceLength + 1):
        BestCost = Inf
        BestIntervalIndex: int | None = None

        for Index, IntervalItem in enumerate(Normalized):
            if IntervalItem.Left <= CoveredUntil <= IntervalItem.Right:
                PreviousCost = 0.0 if IntervalItem.Left == 0 else Dp[IntervalItem.Left - 1]
                if PreviousCost == Inf:
                    continue
                Candidate = PreviousCost + IntervalItem.Cost
                if Candidate < BestCost:
                    BestCost = Candidate
                    BestIntervalIndex = Index
                elif Candidate == BestCost and BestIntervalIndex is not None:
                    CurrentBest = Normalized[BestIntervalIndex]
                    if IntervalItem.Right > CurrentBest.Right:
                        BestIntervalIndex = Index
                    elif IntervalItem.Right == CurrentBest.Right and IntervalItem.Cost < CurrentBest.Cost:
                        BestIntervalIndex = Index

        Dp[CoveredUntil] = BestCost
        Parent[CoveredUntil] = BestIntervalIndex

    if Dp[FenceLength] == Inf:
        return CoverResult(False, Inf, tuple(), "DpExacta")

    Chosen: list[Interval] = []
    Cursor = FenceLength
    while Cursor >= 0:
        IntervalIndex = Parent[Cursor]
        if IntervalIndex is None:
            break
        IntervalItem = Normalized[IntervalIndex]
        Chosen.append(IntervalItem)
        Cursor = IntervalItem.Left - 1

    Chosen.reverse()
    Feasible = CoverIsValid(Chosen, FenceLength)
    return CoverResult(Feasible, SolutionCost(Chosen) if Feasible else Inf, tuple(Chosen), "DpExacta")


def SolveGreedyHeuristic(Intervals: Sequence[Interval], FenceLength: int) -> CoverResult:
    Normalized = NormalizeIntervals(Intervals, FenceLength)

    import heapq as Heapq

    SortedByLeft = sorted(
        enumerate(Normalized),
        key=lambda Item: (Item[1].Left, Item[1].Right, Item[1].Cost, Item[0]),
    )
    Heap: list[tuple[float, int, float, int]] = []
    Chosen: list[Interval] = []
    Cursor = 0
    InsertIndex = 0

    while Cursor <= FenceLength:
        while InsertIndex < len(SortedByLeft) and SortedByLeft[InsertIndex][1].Left <= Cursor:
            Index, IntervalItem = SortedByLeft[InsertIndex]
            Heapq.heappush(Heap, (-IntervalItem.Efficiency, -IntervalItem.Right, IntervalItem.Cost, Index))
            InsertIndex += 1

        while Heap and Normalized[Heap[0][3]].Right < Cursor:
            Heapq.heappop(Heap)

        if not Heap:
            return CoverResult(False, Inf, tuple(), "Greedy")

        _, _, _, Index = Heapq.heappop(Heap)
        IntervalItem = Normalized[Index]
        Chosen.append(IntervalItem)
        Cursor = IntervalItem.Right + 1

    Feasible = CoverIsValid(Chosen, FenceLength)
    return CoverResult(Feasible, SolutionCost(Chosen) if Feasible else Inf, tuple(Chosen), "Greedy")


def GenerateFeasibleInstance(
    FenceLength: int,
    PainterCount: int,
    Rng: Random,
    MaxBackboneStep: int | None = None,
) -> list[Interval]:
    if FenceLength < 0:
        raise ValueError("FenceLength must be non-negative")
    if PainterCount <= 0:
        raise ValueError("PainterCount must be positive")

    Intervals: list[Interval] = []
    BackboneIndex = 0
    Cursor = 0
    MaxBackboneStep = MaxBackboneStep or max(1, FenceLength // 5 or 1)

    while Cursor <= FenceLength and len(Intervals) < PainterCount:
        LeftShift = Rng.randint(0, min(Cursor, 3)) if Cursor > 0 else 0
        Left = max(0, Cursor - LeftShift)
        Step = Rng.randint(1, max(1, min(MaxBackboneStep, FenceLength - Cursor + 1)))
        Right = min(FenceLength, Cursor + Step - 1)
        CostValue = Rng.randint(1, 20)
        Intervals.append(Interval(Left=Left, Right=Right, Cost=float(CostValue), Name=f"b{BackboneIndex}"))
        BackboneIndex += 1
        Cursor = Right + 1

    while len(Intervals) < PainterCount:
        Left = Rng.randint(0, FenceLength)
        Right = Rng.randint(Left, FenceLength)
        CostValue = Rng.randint(1, 25)
        Intervals.append(Interval(Left=Left, Right=Right, Cost=float(CostValue), Name=f"r{len(Intervals)}"))

    return Intervals


def FindGreedyCounterexample(
    FenceLength: int = 12,
    PainterCount: int = 6,
    Seed: int = 7,
    Attempts: int = 50_000,
) -> tuple[list[Interval], CoverResult, CoverResult]:
    Rng = Random(Seed)
    for AttemptIndex in range(Attempts):
        Instance = GenerateFeasibleInstance(FenceLength, PainterCount, Rng)
        Optimal = SolveExactDp(Instance, FenceLength)
        Greedy = SolveGreedyHeuristic(Instance, FenceLength)
        if Optimal.Feasible and Greedy.Feasible and Greedy.TotalCost > Optimal.TotalCost:
            return Instance, Optimal, Greedy

    FallbackInstance = [
        Interval(Left=0, Right=1, Cost=2.0, Name="b0"),
        Interval(Left=1, Right=2, Cost=1.0, Name="b1"),
        Interval(Left=1, Right=3, Cost=9.0, Name="b2"),
    ]
    Optimal = SolveExactDp(FallbackInstance, 3)
    Greedy = SolveGreedyHeuristic(FallbackInstance, 3)
    if Optimal.Feasible and Greedy.Feasible and Greedy.TotalCost > Optimal.TotalCost:
        return FallbackInstance, Optimal, Greedy
    raise RuntimeError("Contraejemplo no encontrado dentro del presupuesto de búsqueda")


def InstanceSize(FenceLength: int, PainterCount: int) -> int:
    return FenceLength + PainterCount


def IntervalsToRows(Intervals: Sequence[Interval]) -> list[tuple[str, int, int, float]]:
    return [(IntervalItem.Name, IntervalItem.Left, IntervalItem.Right, IntervalItem.Cost) for IntervalItem in Intervals]


# --- Subset variant: painters cover arbitrary subsets of discrete sections ---
@dataclass(frozen=True, slots=True)
class PainterSubset:
    Mask: int
    Cost: float
    Name: str = ""


def SolveExactDp_Subsets(Painters: Sequence[PainterSubset], Sections: int) -> CoverResult:
    if Sections < 0:
        raise ValueError("Sections must be non-negative")
    Target = (1 << Sections) - 1
    M = len(Painters)

    InfCost = float('inf')
    Dp = [InfCost] * (1 << Sections)
    Parent: list[tuple[int, int] | None] = [None] * (1 << Sections)
    Dp[0] = 0.0

    for mask in range(1 << Sections):
        if Dp[mask] == InfCost:
            continue
        for idx, painter in enumerate(Painters):
            newmask = mask | painter.Mask
            cost = Dp[mask] + painter.Cost
            if cost < Dp[newmask]:
                Dp[newmask] = cost
                Parent[newmask] = (mask, idx)

    if Dp[Target] == InfCost:
        return CoverResult(False, Inf, tuple(), "DpSubsets")

    # reconstruct
    chosen: list[PainterSubset] = []
    cur = Target
    while cur != 0:
        entry = Parent[cur]
        if entry is None:
            break
        prevmask, idx = entry
        chosen.append(Painters[idx])
        cur = prevmask

    chosen.reverse()
    # Build Intervals-like cover check: convert to covered sections set
    UnionMask = reduce(BitwiseOr, (p.Mask for p in chosen), 0)
    Feasible = (UnionMask & Target) == Target
    return CoverResult(Feasible, sum(p.Cost for p in chosen) if Feasible else Inf, tuple(), "DpSubsets")


def SolveGreedyHeuristic_Subsets(Painters: Sequence[PainterSubset], Sections: int) -> CoverResult:
    if Sections < 0:
        raise ValueError("Sections must be non-negative")
    Target = (1 << Sections) - 1
    Remaining = 0
    Chosen: list[PainterSubset] = []
    Covered = 0
    PaintersList = list(Painters)

    while Covered != Target:
        best_idx = None
        best_score = -1.0
        best_new = 0
        for idx, p in enumerate(PaintersList):
            newbits = p.Mask & (~Covered)
            newcount = newbits.bit_count()
            if newcount == 0:
                continue
            score = newcount / p.Cost
            if score > best_score:
                best_score = score
                best_idx = idx
                best_new = newbits

        if best_idx is None:
            return CoverResult(False, Inf, tuple(), "GreedySubsets")

        p = PaintersList[best_idx]
        Chosen.append(p)
        Covered |= p.Mask

    return CoverResult(True, sum(p.Cost for p in Chosen), tuple(), "GreedySubsets")


def GenerateFeasibleInstance_Subsets(Sections: int, PainterCount: int, Rng: Random) -> list[PainterSubset]:
    if Sections <= 0:
        raise ValueError("Sections must be positive")
    if PainterCount <= 0:
        raise ValueError("PainterCount must be positive")

    Painters: list[PainterSubset] = []
    # backbone: ensure each section has a cheap painter
    for s in range(Sections):
        mask = 1 << s
        cost = float(Rng.randint(1, 3))
        Painters.append(PainterSubset(Mask=mask, Cost=cost, Name=f"b{s}"))

    # add random multi-section painters
    while len(Painters) < PainterCount:
        mask = 0
        for s in range(Sections):
            if Rng.random() < 0.5:
                mask |= 1 << s
        if mask == 0:
            mask = 1 << Rng.randint(0, Sections - 1)
        cost = float(Rng.randint(1, 10))
        Painters.append(PainterSubset(Mask=mask, Cost=cost, Name=f"r{len(Painters)}"))

    # ensure coverage
    union = 0
    for p in Painters:
        union |= p.Mask
    if union != (1 << Sections) - 1:
        missing = ((1 << Sections) - 1) & ~union
        s = 0
        while missing:
            bit = missing & -missing
            missing &= missing - 1
            Painters.append(PainterSubset(Mask=bit, Cost=1.0, Name=f"fix{s}"))
            s += 1

    return Painters
