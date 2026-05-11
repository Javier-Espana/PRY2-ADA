"""Benchmark runner for the Paint the Fence project.

It generates reproducible random instances, measures the exact DP and the greedy
heuristic, fits polynomial regressions, and writes LaTeX/CSV artifacts into results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import sys
import time
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paint_the_fence import (  # noqa: E402
    find_greedy_counterexample,
    generate_feasible_instance,
    instance_size,
    solve_exact_dp,
    solve_greedy_heuristic,
)


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _time_call(function, *args):
    start = time.perf_counter()
    result = function(*args)
    end = time.perf_counter()
    return result, (end - start) * 1000.0


def _fit_best_polynomial(x_values: np.ndarray, y_values: np.ndarray, max_degree: int = 4):
    best = None
    for degree in range(1, min(max_degree, len(x_values) - 1) + 1):
        coefficients = np.polyfit(x_values, y_values, degree)
        polynomial = np.poly1d(coefficients)
        prediction = polynomial(x_values)
        residuals = y_values - prediction
        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
        r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
        candidate = {
            "degree": degree,
            "coefficients": coefficients,
            "polynomial": polynomial,
            "mse": mse,
            "rmse": rmse,
            "r_squared": r_squared,
        }
        if best is None or candidate["rmse"] < best["rmse"]:
            best = candidate
    if best is None:
        raise RuntimeError("no polynomial fit could be computed")
    return best


def _format_polynomial(coefficients: np.ndarray) -> str:
    degree = len(coefficients) - 1
    pieces = []
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        value = float(coefficient)
        if abs(value) < 1e-12:
            continue
        sign = "+" if value >= 0 else "-"
        magnitude = abs(value)
        if power == 0:
            term = f"{magnitude:.6g}"
        elif power == 1:
            term = f"{magnitude:.6g}x"
        else:
            term = f"{magnitude:.6g}x^{power}"
        if not pieces:
            pieces.append(term if value >= 0 else f"-{term}")
        else:
            pieces.append(f" {sign} {term}")
    return "".join(pieces) if pieces else "0"


def _write_csv(rows, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_plot(summary_rows, dp_fit, greedy_fit, output_path: Path) -> None:
    x = np.array([row["input_size"] for row in summary_rows], dtype=float)
    dp_y = np.array([row["dp_median_ms"] for row in summary_rows], dtype=float)
    greedy_y = np.array([row["greedy_median_ms"] for row in summary_rows], dtype=float)

    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 400)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, dp_y, color="#1f77b4", label="DP exacta", s=45)
    ax.scatter(x, greedy_y, color="#d62728", label="Greedy", s=45)
    ax.plot(x_dense, dp_fit["polynomial"](x_dense), color="#1f77b4", alpha=0.8)
    ax.plot(x_dense, greedy_fit["polynomial"](x_dense), color="#d62728", alpha=0.8)
    ax.set_xlabel("Tamano de la entrada (L + n)")
    ax.set_ylabel("Tiempo medio de ejecucion (ms)")
    ax.set_title("Cobertura de la valla: comparacion de tiempos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_benchmark(fence_lengths, repetitions: int, seed: int):
    rows = []
    summary_rows = []
    for fence_length in fence_lengths:
        painter_count = fence_length * 2
        dp_times = []
        greedy_times = []
        ratios = []
        for repetition in range(repetitions):
            instance_seed = seed + fence_length * 1000 + repetition
            random_instance = random.Random(instance_seed)
            intervals = generate_feasible_instance(fence_length, painter_count, random_instance)

            exact, exact_ms = _time_call(solve_exact_dp, intervals, fence_length)
            greedy, greedy_ms = _time_call(solve_greedy_heuristic, intervals, fence_length)

            if not exact.feasible:
                raise RuntimeError(f"exact solver failed on fence_length={fence_length}")
            if not greedy.feasible:
                raise RuntimeError(f"greedy solver failed on fence_length={fence_length}")

            ratio = greedy.total_cost / exact.total_cost if exact.total_cost else float("nan")
            rows.append(
                {
                    "fence_length": fence_length,
                    "painter_count": painter_count,
                    "input_size": instance_size(fence_length, painter_count),
                    "repetition": repetition,
                    "exact_ms": exact_ms,
                    "greedy_ms": greedy_ms,
                    "optimal_cost": exact.total_cost,
                    "greedy_cost": greedy.total_cost,
                    "approximation_ratio": ratio,
                }
            )
            dp_times.append(exact_ms)
            greedy_times.append(greedy_ms)
            ratios.append(ratio)

        summary_rows.append(
            {
                "fence_length": fence_length,
                "painter_count": painter_count,
                "input_size": instance_size(fence_length, painter_count),
                "dp_median_ms": float(stats.median(dp_times)),
                "greedy_median_ms": float(stats.median(greedy_times)),
                "ratio_mean": float(stats.mean(ratios)),
                "ratio_max": float(max(ratios)),
            }
        )

    return rows, summary_rows


def _write_summary_tex(summary_rows, dp_fit, greedy_fit, counterexample, path: Path) -> None:
    counterexample_instance, optimal, greedy = counterexample
    interval_rows = []
    for interval in counterexample_instance:
        interval_rows.append(f"{interval.name or '-'} & {interval.left} & {interval.right} & {interval.cost:.0f} \\\\")

    table_rows = []
    for row in summary_rows:
        table_rows.append(
            f"{row['fence_length']} & {row['painter_count']} & {row['input_size']} & "
            f"{row['dp_median_ms']:.4f} & {row['greedy_median_ms']:.4f} & {row['ratio_mean']:.4f} \\\\")

    content = rf"""% Auto-generated by scripts/benchmark.py
\newcommand{{\DPPolynomial}}{{\ensuremath{{{_format_polynomial(dp_fit['coefficients'])}}}}}
\newcommand{{\GreedyPolynomial}}{{\ensuremath{{{_format_polynomial(greedy_fit['coefficients'])}}}}}
\newcommand{{\DPDegree}}{{{dp_fit['degree']}}}
\newcommand{{\GreedyDegree}}{{{greedy_fit['degree']}}}
\newcommand{{\DPRSquared}}{{{dp_fit['r_squared']:.4f}}}
\newcommand{{\GreedyRSquared}}{{{greedy_fit['r_squared']:.4f}}}
\newcommand{{\DPRMSE}}{{{dp_fit['rmse']:.4f}}}
\newcommand{{\GreedyRMSE}}{{{greedy_fit['rmse']:.4f}}}
\newcommand{{\MeanApproxRatio}}{{{float(stats.mean([row['ratio_mean'] for row in summary_rows])):.4f}}}
\newcommand{{\WorstApproxRatio}}{{{float(max(row['ratio_max'] for row in summary_rows)):.4f}}}

\begin{{tabular}}{{rrrrrr}}
\toprule
$L$ & $n$ & $L+n$ & DP med. (ms) & Greedy med. (ms) & Raz\'on media \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}

\paragraph{{Counterexample usado en la discusion de greedy choice property.}} La instancia encontrada por busqueda aleatoria contiene la siguiente lista de intervalos:

\begin{{tabular}}{{rrrr}}
\toprule
Nombre & Izquierda & Derecha & Costo \\
\midrule
{chr(10).join(interval_rows)}
\bottomrule
\end{{tabular}}

La solucion optima exacta cuesta {optimal.total_cost:.0f} y la heuristica greedy cuesta {greedy.total_cost:.0f}.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=800)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()

    fence_lengths = list(range(args.min_length, args.max_length + 1, args.step))
    rows, summary_rows = run_benchmark(fence_lengths, args.repetitions, args.seed)

    csv_path = RESULTS_DIR / "benchmark.csv"
    summary_csv_path = RESULTS_DIR / "benchmark_summary.csv"
    plot_path = RESULTS_DIR / "benchmark_scatter.png"
    summary_tex_path = RESULTS_DIR / "benchmark_summary.tex"
    counterexample_json_path = RESULTS_DIR / "counterexample.json"

    _write_csv(rows, csv_path)
    _write_csv(summary_rows, summary_csv_path)

    x = np.array([row["input_size"] for row in summary_rows], dtype=float)
    dp_y = np.array([row["dp_median_ms"] for row in summary_rows], dtype=float)
    greedy_y = np.array([row["greedy_median_ms"] for row in summary_rows], dtype=float)
    dp_fit = _fit_best_polynomial(x, dp_y)
    greedy_fit = _fit_best_polynomial(x, greedy_y)

    counterexample = find_greedy_counterexample()
    instance, optimal, greedy = counterexample
    counterexample_fence_length = max(interval.right for interval in instance)
    counterexample_json_path.write_text(
        json.dumps(
            {
                "fence_length": counterexample_fence_length,
                "intervals": [
                    {"left": interval.left, "right": interval.right, "cost": interval.cost, "name": interval.name}
                    for interval in instance
                ],
                "optimal_cost": optimal.total_cost,
                "greedy_cost": greedy.total_cost,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _make_plot(summary_rows, dp_fit, greedy_fit, plot_path)
    _write_summary_tex(summary_rows, dp_fit, greedy_fit, counterexample, summary_tex_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_tex_path}")
    print(f"Wrote {counterexample_json_path}")
    print(f"DP fit: degree {dp_fit['degree']}, R^2={dp_fit['r_squared']:.4f}")
    print(f"Greedy fit: degree {greedy_fit['degree']}, R^2={greedy_fit['r_squared']:.4f}")


if __name__ == "__main__":
    main()
