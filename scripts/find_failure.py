import sys
import random
from pathlib import Path
Root = Path(__file__).resolve().parents[1]
if str(Root) not in sys.path:
    sys.path.insert(0, str(Root))

from PaintTheFence import GenerateFeasibleInstance_Subsets, SolveExactDp_Subsets
Sections = 4
for s in range(500):
    R = random.Random(s)
    P = GenerateFeasibleInstance_Subsets(Sections, 8, R)
    Res = SolveExactDp_Subsets(P, Sections)
    if not Res.Feasible:
        print(f"FAILED Seed {s}")
        exit(0)
print("No failure in 500 seeds")
