import argparse
import random
import sys

from src.PaintTheFence import (
    GenerateFeasibleInstance_Subsets,
    SolveExactDp_Subsets,
    SolveGreedyHeuristic_Subsets,
)


def main():
    parser = argparse.ArgumentParser(description="Paint The Fence - Proyecto #2 ADA")
    parser.add_argument("--fence-length", type=int, default=20, help="Número de secciones")
    parser.add_argument("--painters", type=int, default=10, help="Cantidad de pintores")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para la generación aleatoria")
    parser.add_argument("--benchmark", action="store_true", help="Ejecuta los benchmarks de subsets")

    args = parser.parse_args()

    if args.benchmark:
        print("Ejecutando benchmarks de subsets (esto puede tardar unos minutos)...")
        try:
            from scripts import benchmark
            sys.argv = [sys.argv[0]]
            benchmark.Main()
            print("Benchmarks finalizados. Revisa la carpeta results/ para los resultados de subsets.")
        except ImportError as e:
            print(f"Error al importar el benchmark: {e}")
            sys.exit(1)
        return

    print(f"--- Paint the Fence (Secciones={args.fence_length}, Pintores={args.painters}, Semilla={args.seed}) ---")
    rng = random.Random(args.seed)

    try:
        instance = GenerateFeasibleInstance_Subsets(args.fence_length, args.painters, rng)
    except ValueError as e:
        print(f"Error al generar la instancia: {e}")
        sys.exit(1)

    print("\nPintores generados:")
    for painter in instance:
        print(f"  {painter.Name}: mascara={bin(painter.Mask)}, Costo: {painter.Cost:.2f}")

    print("\nCalculando solución exacta (Programación Dinámica sobre subsets)...")
    dp_res = SolveExactDp_Subsets(instance, args.fence_length)
    if dp_res.Feasible:
        print(f"  [DP] Costo óptimo: {dp_res.TotalCost:.2f}")
        print(f"  [DP] Pintores seleccionados: {', '.join(p.Name for p in dp_res.Selected)}")
    else:
        print("  [DP] No se encontró una solución factible.")

    print("\nCalculando solución heurística (Greedy)...")
    greedy_res = SolveGreedyHeuristic_Subsets(instance, args.fence_length)
    if greedy_res.Feasible:
        print(f"  [Greedy] Costo heurístico: {greedy_res.TotalCost:.2f}")
        print(f"  [Greedy] Pintores seleccionados: {', '.join(p.Name for p in greedy_res.Selected)}")
    else:
        print("  [Greedy] No se encontró una solución factible.")

    if dp_res.Feasible and greedy_res.Feasible:
        ratio = greedy_res.TotalCost / dp_res.TotalCost
        print(f"\nRazón de aproximación (Greedy/DP): {ratio:.4f}")


if __name__ == '__main__':
    main()
