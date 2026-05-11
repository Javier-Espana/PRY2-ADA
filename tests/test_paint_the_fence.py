from __future__ import annotations

import unittest

from paint_the_fence import (
    Interval,
    cover_is_valid,
    find_greedy_counterexample,
    solve_exact_dp,
    solve_greedy_heuristic,
)


class PaintTheFenceTests(unittest.TestCase):
    def test_exact_dp_finds_optimal_cover(self):
        intervals = [
            Interval(0, 1, 2, "a"),
            Interval(1, 3, 3, "b"),
        ]
        result = solve_exact_dp(intervals, 3)
        self.assertTrue(result.feasible)
        self.assertEqual(result.total_cost, 5.0)
        self.assertTrue(cover_is_valid(result.intervals, 3))

    def test_greedy_returns_a_valid_cover_when_possible(self):
        intervals = [
            Interval(0, 2, 3, "a"),
            Interval(1, 4, 3, "b"),
            Interval(4, 6, 2, "c"),
            Interval(0, 6, 20, "d"),
        ]
        result = solve_greedy_heuristic(intervals, 6)
        self.assertTrue(result.feasible)
        self.assertTrue(cover_is_valid(result.intervals, 6))
        self.assertGreater(result.total_cost, 0)

    def test_counterexample_exists(self):
        _, optimal, greedy = find_greedy_counterexample()
        self.assertTrue(optimal.feasible)
        self.assertTrue(greedy.feasible)
        self.assertGreater(greedy.total_cost, optimal.total_cost)
        self.assertTrue(cover_is_valid(optimal.intervals, 3))
        self.assertTrue(cover_is_valid(greedy.intervals, 3))
