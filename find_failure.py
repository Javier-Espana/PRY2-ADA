from PaintTheFence import GenerateFeasibleInstance_Subsets, SolveExactDp_Subsets
import random
Sections = 4
for s in range(500):
    R = random.Random(s)
    P = GenerateFeasibleInstance_Subsets(Sections, 8, R)
    Res = SolveExactDp_Subsets(P, Sections)
    if not Res.Feasible:
        print(f"FAILED Seed {s}")
        exit(0)
print("No failure in 500 seeds")
