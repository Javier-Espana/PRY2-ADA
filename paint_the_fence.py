"""Exact and greedy solutions for the Paint the Fence interval cover problem.

The problem is modeled on a discrete fence with positions from 0 to L.
Each painter covers a closed integer interval [left, right] with an associated cost.
The exact dynamic-programming solution is pseudo-polynomial in the fence length.
The greedy solution is a cost-efficiency heuristic used for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from random import Random
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Interval:
    """A painter available for a single interval of the fence."""

    left: int
    right: int
    cost: float
    name: str = ""

    @property
    def length(self) -> int:
        return self.right - self.left + 1

    @property
    def efficiency(self) -> float:
        return self.length / self.cost


@dataclass(frozen=True, slots=True)
class CoverResult:
    """Stores the outcome of a covering algorithm."""

    feasible: bool
    total_cost: float
    intervals: tuple[Interval, ...]
    algorithm: str


def normalize_intervals(intervals: Iterable[Interval], fence_length: int) -> list[Interval]:
    """Clip intervals to the fence and discard irrelevant ones."""

    if fence_length < 0:
        raise ValueError("fence_length must be non-negative")

    normalized: list[Interval] = []
    for interval in intervals:
        if interval.cost <= 0:
            raise ValueError("interval costs must be positive")
        left = max(0, min(fence_length, interval.left))
        right = max(0, min(fence_length, interval.right))
        if left > right:
            continue
        normalized.append(Interval(left=left, right=right, cost=float(interval.cost), name=interval.name))
    return normalized


def cover_is_valid(intervals: Sequence[Interval], fence_length: int) -> bool:
    """Check whether the selected intervals cover the whole fence."""

    if fence_length < 0:
        return False
    sorted_intervals = sorted(intervals, key=lambda interval: (interval.left, interval.right, interval.cost))
    next_uncovered = 0
    for interval in sorted_intervals:
        if interval.right < next_uncovered:
            continue
        if interval.left > next_uncovered:
            return False
        next_uncovered = interval.right + 1
        if next_uncovered > fence_length:
            return True
    return next_uncovered > fence_length


def solution_cost(intervals: Sequence[Interval]) -> float:
    return sum(interval.cost for interval in intervals)


def solve_exact_dp(intervals: Sequence[Interval], fence_length: int) -> CoverResult:
    """Exact pseudo-polynomial dynamic program.

    dp[x] is the minimum cost to cover the prefix [0, x].
    The recurrence is:

        dp[x] = min(dp[left_i - 1] + cost_i)

    over all intervals i such that left_i <= x <= right_i.
    """

    normalized = normalize_intervals(intervals, fence_length)
    if fence_length == 0:
        return CoverResult(True, 0.0, tuple(), "dp_exacta")

    dp = [inf] * (fence_length + 1)
    parent: list[int | None] = [None] * (fence_length + 1)

    for covered_until in range(fence_length + 1):
        best_cost = inf
        best_interval_index: int | None = None

        for index, interval in enumerate(normalized):
            if interval.left <= covered_until <= interval.right:
                previous_cost = 0.0 if interval.left == 0 else dp[interval.left - 1]
                if previous_cost == inf:
                    continue
                candidate = previous_cost + interval.cost
                if candidate < best_cost:
                    best_cost = candidate
                    best_interval_index = index
                elif candidate == best_cost and best_interval_index is not None:
                    current_best = normalized[best_interval_index]
                    if interval.right > current_best.right:
                        best_interval_index = index
                    elif interval.right == current_best.right and interval.cost < current_best.cost:
                        best_interval_index = index

        dp[covered_until] = best_cost
        parent[covered_until] = best_interval_index

    if dp[fence_length] == inf:
        return CoverResult(False, inf, tuple(), "dp_exacta")

    chosen: list[Interval] = []
    cursor = fence_length
    while cursor >= 0:
        interval_index = parent[cursor]
        if interval_index is None:
            break
        interval = normalized[interval_index]
        chosen.append(interval)
        cursor = interval.left - 1

    chosen.reverse()
    feasible = cover_is_valid(chosen, fence_length)
    return CoverResult(feasible, solution_cost(chosen) if feasible else inf, tuple(chosen), "dp_exacta")


def solve_greedy_heuristic(intervals: Sequence[Interval], fence_length: int) -> CoverResult:
    """Greedy heuristic based on coverage-per-cost efficiency.

    At each uncovered position, it chooses among the available intervals the one
    with the largest static efficiency length / cost, breaking ties by farther reach.
    """

    normalized = normalize_intervals(intervals, fence_length)
    if fence_length == 0:
        return CoverResult(True, 0.0, tuple(), "greedy")

    import heapq

    sorted_by_left = sorted(enumerate(normalized), key=lambda item: (item[1].left, item[1].right, item[1].cost, item[0]))
    heap: list[tuple[float, int, float, int]] = []
    chosen: list[Interval] = []
    cursor = 0
    insert_index = 0

    while cursor <= fence_length:
        while insert_index < len(sorted_by_left) and sorted_by_left[insert_index][1].left <= cursor:
            index, interval = sorted_by_left[insert_index]
            heapq.heappush(heap, (-interval.efficiency, -interval.right, interval.cost, index))
            insert_index += 1

        while heap and normalized[heap[0][3]].right < cursor:
            heapq.heappop(heap)

        if not heap:
            return CoverResult(False, inf, tuple(), "greedy")

        _, _, _, index = heapq.heappop(heap)
        interval = normalized[index]
        chosen.append(interval)
        cursor = interval.right + 1

    feasible = cover_is_valid(chosen, fence_length)
    return CoverResult(feasible, solution_cost(chosen) if feasible else inf, tuple(chosen), "greedy")


def generate_feasible_instance(
    fence_length: int,
    painter_count: int,
    rng: Random,
    max_backbone_step: int | None = None,
) -> list[Interval]:
    """Generate a random feasible instance for benchmarking.

    The instance is guaranteed to be coverable by creating a backbone of intervals.
    Additional random intervals are added to increase the search space.
    """

    if fence_length < 0:
        raise ValueError("fence_length must be non-negative")
    if painter_count <= 0:
        raise ValueError("painter_count must be positive")

    intervals: list[Interval] = []
    backbone_index = 0
    cursor = 0
    max_backbone_step = max_backbone_step or max(1, fence_length // 5 or 1)

    while cursor <= fence_length and len(intervals) < painter_count:
        left_shift = rng.randint(0, min(cursor, 3)) if cursor > 0 else 0
        left = max(0, cursor - left_shift)
        step = rng.randint(1, max(1, min(max_backbone_step, fence_length - cursor + 1)))
        right = min(fence_length, cursor + step - 1)
        cost = rng.randint(1, 20)
        intervals.append(Interval(left=left, right=right, cost=float(cost), name=f"b{backbone_index}"))
        backbone_index += 1
        cursor = right + 1

    while len(intervals) < painter_count:
        left = rng.randint(0, fence_length)
        right = rng.randint(left, fence_length)
        cost = rng.randint(1, 25)
        intervals.append(Interval(left=left, right=right, cost=float(cost), name=f"r{len(intervals)}"))

    return intervals


def find_greedy_counterexample(
    fence_length: int = 12,
    painter_count: int = 6,
    seed: int = 7,
    attempts: int = 50_000,
) -> tuple[list[Interval], CoverResult, CoverResult]:
    """Search for an instance where greedy is worse than the exact DP."""

    rng = Random(seed)
    for _ in range(attempts):
        instance = generate_feasible_instance(fence_length, painter_count, rng)
        optimal = solve_exact_dp(instance, fence_length)
        greedy = solve_greedy_heuristic(instance, fence_length)
        if optimal.feasible and greedy.feasible and greedy.total_cost > optimal.total_cost:
            return instance, optimal, greedy

    fallback_instance = [
        Interval(0, 1, 2.0, "b0"),
        Interval(1, 2, 1.0, "b1"),
        Interval(1, 3, 9.0, "b2"),
    ]
    optimal = solve_exact_dp(fallback_instance, 3)
    greedy = solve_greedy_heuristic(fallback_instance, 3)
    if optimal.feasible and greedy.feasible and greedy.total_cost > optimal.total_cost:
        return fallback_instance, optimal, greedy
    raise RuntimeError("counterexample not found within the search budget")


def instance_size(fence_length: int, painter_count: int) -> int:
    return fence_length + painter_count


def intervals_to_rows(intervals: Sequence[Interval]) -> list[tuple[str, int, int, float]]:
    return [(interval.name, interval.left, interval.right, interval.cost) for interval in intervals]
