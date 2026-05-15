from __future__ import annotations

import sys as Sys
import unittest as UnitTest
from pathlib import Path

Root = Path(__file__).resolve().parents[1]
if str(Root) not in Sys.path:
    Sys.path.insert(0, str(Root))

from src.PaintTheFence import (
    Interval,
    CoverIsValid,
    FindGreedyCounterexample,
    NormalizeIntervals,
    SolveExactDp,
    SolveGreedyHeuristic,
    PainterSubset,
    SolveExactDp_Subsets,
    SolveGreedyHeuristic_Subsets,
    GenerateFeasibleInstance_Subsets,
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

    def TestSubsetDpFindsOptimal(self):
        # Sections: 3 (bits 0..2)
        Painters = [
            PainterSubset(Mask=0b001, Cost=1.0, Name="a"),
            PainterSubset(Mask=0b010, Cost=1.0, Name="b"),
            PainterSubset(Mask=0b100, Cost=1.0, Name="c"),
            PainterSubset(Mask=0b111, Cost=5.0, Name="big"),
        ]
        Result = SolveExactDp_Subsets(Painters, 3)
        self.assertTrue(Result.Feasible)
        self.assertEqual(Result.TotalCost, 3.0)

    def TestSubsetGreedyValid(self):
        Rng = __import__('random').Random(42)
        Painters = GenerateFeasibleInstance_Subsets(5, 10, Rng)
        Greedy = SolveGreedyHeuristic_Subsets(Painters, 5)
        self.assertTrue(Greedy.Feasible)
