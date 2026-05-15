import sys
from pathlib import Path
Root = Path(__file__).resolve().parents[1]
if str(Root) not in sys.path:
    sys.path.insert(0, str(Root))

from PaintTheFence import SolveExactDp_Subsets, PainterSubset
P = [
    PainterSubset(Mask=0b1, Cost=2.0, Name='b0'),
    PainterSubset(Mask=0b10, Cost=1.0, Name='b1'),
    PainterSubset(Mask=0b100, Cost=2.0, Name='b2'),
    PainterSubset(Mask=0b1000, Cost=3.0, Name='b3')
]
Res = SolveExactDp_Subsets(P, 4)
print(f"Manual check: Mask 1+2+4+8 = 15. Feasible: {Res.Feasible}")
