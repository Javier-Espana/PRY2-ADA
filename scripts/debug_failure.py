import sys
import random
from pathlib import Path
Root = Path(__file__).resolve().parents[1]
if str(Root) not in sys.path:
    sys.path.insert(0, str(Root))

from PaintTheFence import GenerateFeasibleInstance_Subsets, SolveExactDp_Subsets
R = random.Random(47)
Sections = 4
P = GenerateFeasibleInstance_Subsets(Sections, 8, R)
print(f"Painters:")
for p in P:
    print(f"Mask: {bin(p.Mask)}, Cost: {p.Cost}, Name: {p.Name}")
Res = SolveExactDp_Subsets(P, Sections)
print(f"Feasible: {Res.Feasible}, Cost: {Res.TotalCost}")
