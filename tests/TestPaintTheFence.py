from __future__ import annotations

import unittest as UnitTest

from PaintTheFence import (
    Interval,
    CoverIsValid,
    FindGreedyCounterexample,
    NormalizeIntervals,
    SolveExactDp,
    SolveGreedyHeuristic,
)


DefaultTestLoader = UnitTest.defaultTestLoader
DefaultTestLoader.testMethodPrefix = "Test"


class PaintTheFenceTests(UnitTest.TestCase):
    def TestExactDpFindsOptimalCover(self):
        Intervals = [
            Interval(0, 1, 2, "a"),
            Interval(1, 3, 3, "b"),
        ]
        Result = SolveExactDp(Intervals, 3)
        self.assertTrue(Result.Feasible)
        self.assertEqual(Result.TotalCost, 5.0)
        self.assertTrue(CoverIsValid(Result.Intervals, 3))

    def TestGreedyReturnsAValidCoverWhenPossible(self):
        Intervals = [
            Interval(0, 2, 3, "a"),
            Interval(1, 4, 3, "b"),
            Interval(4, 6, 2, "c"),
            Interval(0, 6, 20, "d"),
        ]
        Result = SolveGreedyHeuristic(Intervals, 6)
        self.assertTrue(Result.Feasible)
        self.assertTrue(CoverIsValid(Result.Intervals, 6))
        self.assertGreater(Result.TotalCost, 0)

    def TestCounterexampleExists(self):
        _, Optimal, Greedy = FindGreedyCounterexample()
        self.assertTrue(Optimal.Feasible)
        self.assertTrue(Greedy.Feasible)
        self.assertGreater(Greedy.TotalCost, Optimal.TotalCost)
        FenceLength = max(IntervalItem.Right for IntervalItem in Optimal.Intervals)
        self.assertTrue(CoverIsValid(Optimal.Intervals, FenceLength))
        self.assertTrue(CoverIsValid(Greedy.Intervals, FenceLength))

    def TestNormalizeIntervalsDiscardsOutside(self):
        Intervals = [
            Interval(-5, -1, 2, "a"),
            Interval(5, 7, 3, "b"),
            Interval(-2, 1, 4, "c"),
            Interval(2, 5, 5, "d"),
        ]
        Normalized = NormalizeIntervals(Intervals, 3)
        self.assertEqual([(IntervalItem.Left, IntervalItem.Right) for IntervalItem in Normalized], [(0, 1), (2, 3)])

    def TestZeroLengthFenceRequiresCover(self):
        Empty = []
        DpEmpty = SolveExactDp(Empty, 0)
        GreedyEmpty = SolveGreedyHeuristic(Empty, 0)
        self.assertFalse(DpEmpty.Feasible)
        self.assertFalse(GreedyEmpty.Feasible)

        Intervals = [Interval(0, 0, 1, "a")]
        DpCover = SolveExactDp(Intervals, 0)
        GreedyCover = SolveGreedyHeuristic(Intervals, 0)
        self.assertTrue(DpCover.Feasible)
        self.assertTrue(GreedyCover.Feasible)
        self.assertTrue(CoverIsValid(DpCover.Intervals, 0))
        self.assertTrue(CoverIsValid(GreedyCover.Intervals, 0))
