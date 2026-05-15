import gc
import time
import random
from src.PaintTheFence import GenerateFeasibleInstance, SolveExactDp

def test_L(L):
    # Try different repetitions to see variance
    times = []
    for r in range(15):
        rng = random.Random(2026 + L * 1000 + r)
        intervals = GenerateFeasibleInstance(L, L*2, rng)
        
        gc.collect()
        t0 = time.perf_counter()
        SolveExactDp(intervals, L)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    # Return median
    return times[len(times)//2]

print(f"L=900 (Input 2700): {test_L(900):.2f} ms")
print(f"L=950 (Input 2850): {test_L(950):.2f} ms")
print(f"L=1000 (Input 3000): {test_L(1000):.2f} ms")
