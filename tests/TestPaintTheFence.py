from __future__ import annotations

import sys as Sys
import unittest as UnitTest
from pathlib import Path

Root = Path(__file__).resolve().parents[1]
if str(Root) not in Sys.path:
    Sys.path.insert(0, str(Root))

from src.PaintTheFence import (
    PainterSubset,
    SolveExactDp_Subsets,
    SolveGreedyHeuristic_Subsets,
    GenerateFeasibleInstance_Subsets,
)


DefaultTestLoader = UnitTest.defaultTestLoader
DefaultTestLoader.testMethodPrefix = "Test"


class PaintTheFenceTests(UnitTest.TestCase):
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

    def TestSubsetInstanceAlwaysCoversAllSections(self):
        Rng = __import__('random').Random(7)
        Painters = GenerateFeasibleInstance_Subsets(6, 12, Rng)
        UnionMask = 0
        for Painter in Painters:
            UnionMask |= Painter.Mask
        self.assertEqual(UnionMask, (1 << 6) - 1)
