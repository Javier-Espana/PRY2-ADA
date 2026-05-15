from PaintTheFence import GenerateFeasibleInstance_Subsets, SolveExactDp_Subsets
import random
R = random.Random(47)
Sections = 4
P = GenerateFeasibleInstance_Subsets(Sections, 8, R)
print(f"Painters:")
for p in P:
    print(f"Mask: {bin(p.Mask)}, Cost: {p.Cost}, Name: {p.Name}")
Res = SolveExactDp_Subsets(P, Sections)
print(f"Feasible: {Res.Feasible}, Cost: {Res.TotalCost}")
