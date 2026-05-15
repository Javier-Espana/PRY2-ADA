import argparse
import sys
from pathlib import Path

from PaintTheFence import (
    GenerateFeasibleInstance,
    SolveExactDp,
    SolveGreedyHeuristic,
    Interval
)

def main():
    parser = argparse.ArgumentParser(description="Paint The Fence - Proyecto #2 ADA")
    parser.add_argument("--fence-length", type=int, default=20, help="Longitud de la valla")
    parser.add_argument("--painters", type=int, default=10, help="Cantidad de pintores")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para la generación aleatoria")
    parser.add_argument("--benchmark", action="store_true", help="Ejecuta los benchmarks completos")
    
    args = parser.parse_args()

    if args.benchmark:
        print("Ejecutando benchmarks completos (esto puede tardar unos minutos)...")
        # Add scripts to path to import benchmark
        scripts_dir = Path(__file__).parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            import benchmark
            # Reset sys.argv so benchmark's argparse doesn't get confused
            sys.argv = [sys.argv[0]]
            benchmark.Main()
            print("Benchmarks finalizados. Revisa la carpeta results/ para los resultados.")
        except ImportError as e:
            print(f"Error al importar el benchmark: {e}")
            sys.exit(1)
        return

    print(f"--- Paint the Fence (L={args.fence_length}, Pintores={args.painters}, Semilla={args.seed}) ---")
    import random
    rng = random.Random(args.seed)
    
    try:
        instance = GenerateFeasibleInstance(args.fence_length, args.painters, rng)
    except ValueError as e:
        print(f"Error al generar la instancia: {e}")
        sys.exit(1)
        
    print("\nIntervalos generados:")
    for i in instance:
        print(f"  {i.Name}: [{i.Left}, {i.Right}], Costo: {i.Cost:.2f}")
        
    print("\nCalculando solución exacta (Programación Dinámica)...")
    dp_res = SolveExactDp(instance, args.fence_length)
    if dp_res.Feasible:
        print(f"  [DP] Costo óptimo: {dp_res.TotalCost:.2f}")
        print(f"  [DP] Intervalos usados: {', '.join(i.Name for i in dp_res.Intervals)}")
    else:
        print("  [DP] No se encontró una solución factible.")
        
    print("\nCalculando solución heurística (Greedy)...")
    greedy_res = SolveGreedyHeuristic(instance, args.fence_length)
    if greedy_res.Feasible:
        print(f"  [Greedy] Costo heurístico: {greedy_res.TotalCost:.2f}")
        print(f"  [Greedy] Intervalos usados: {', '.join(i.Name for i in greedy_res.Intervals)}")
    else:
        print("  [Greedy] No se encontró una solución factible.")

    if dp_res.Feasible and greedy_res.Feasible:
        ratio = greedy_res.TotalCost / dp_res.TotalCost
        print(f"\nRazón de aproximación (Greedy/DP): {ratio:.4f}")

if __name__ == '__main__':
    main()
