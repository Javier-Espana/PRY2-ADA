from __future__ import annotations

import argparse as Argparse
import csv as Csv
import json as Json
import statistics as Stats
import gc
import sys as Sys
import time as Time
import random as RandomModule
from pathlib import Path

import matplotlib.pyplot as Plt
import numpy as Np

Root = Path(__file__).resolve().parents[1]
if str(Root) not in Sys.path:
    Sys.path.insert(0, str(Root))

from src.PaintTheFence import (
    FindGreedyCounterexample,
    GenerateFeasibleInstance,
    InstanceSize,
    SolveExactDp,
    SolveGreedyHeuristic,
    PainterSubset,
    GenerateFeasibleInstance_Subsets,
    SolveExactDp_Subsets,
    SolveGreedyHeuristic_Subsets,
)

ResultsDir = Root / "results"
ResultsDir.mkdir(exist_ok=True)


def _TimeCall(Function, *Args):
    # Warm-up to avoid cold-start artifacts, then force GC before timing
    try:
        Function(*Args)
    except Exception:
        pass
    gc.collect()
    Start = Time.perf_counter()
    Result = Function(*Args)
    End = Time.perf_counter()
    return Result, (End - Start) * 1000.0


def _FitBestPolynomial(XValues: Np.ndarray, YValues: Np.ndarray, MaxDegree: int = 4):
    """Select polynomial degree by minimising BIC (Bayesian Information Criterion).

    BIC = n·ln(MSE) + k·ln(n), where k = degree+1 (number of fitted parameters).
    BIC penalises extra parameters more strongly than adjusted R², which prevents
    overfitting on small samples and favours the theoretically parsimonious degree.
    """
    import math as _Math

    Best = None
    N = len(XValues)
    LogN = _Math.log(N)
    for Degree in range(1, min(MaxDegree, N - 1) + 1):
        Coefficients = Np.polyfit(XValues, YValues, Degree)
        Polynomial = Np.poly1d(Coefficients)
        Prediction = Polynomial(XValues)
        Residuals = YValues - Prediction
        Mse = float(Np.mean(Residuals**2))
        Rmse = float(Np.sqrt(Mse))
        SsRes = float(Np.sum(Residuals**2))
        SsTot = float(Np.sum((YValues - Np.mean(YValues)) ** 2))
        RSquared = 1.0 if SsTot == 0 else 1.0 - SsRes / SsTot
        AdjRSquared = (
            1.0 - (1.0 - RSquared) * (N - 1) / max(1, N - Degree - 1)
            if SsTot != 0 else 1.0
        )
        # BIC: lower is better; a higher degree wins only when MSE drops enough
        # to offset the ln(n) penalty per added parameter.
        Bic = (
            N * _Math.log(max(Mse, 1e-300)) + (Degree + 1) * LogN
            if Mse > 0 else -float("inf")
        )
        Candidate = {
            "Degree": Degree,
            "Coefficients": Coefficients,
            "Polynomial": Polynomial,
            "Mse": Mse,
            "Rmse": Rmse,
            "RSquared": RSquared,
            "AdjRSquared": AdjRSquared,
            "BIC": Bic,
        }
        if Best is None or Candidate["BIC"] < Best["BIC"]:
            Best = Candidate
    if Best is None:
        raise RuntimeError("no polynomial fit could be computed")
    return Best


def _FormatPolynomial(Coefficients: Np.ndarray) -> str:
    Degree = len(Coefficients) - 1
    Pieces = []
    for Index, Coefficient in enumerate(Coefficients):
        Power = Degree - Index
        Value = float(Coefficient)
        if abs(Value) < 1e-12:
            continue
        Sign = "+" if Value >= 0 else "-"
        Magnitude = abs(Value)
        if Power == 0:
            Term = f"{Magnitude:.6g}"
        elif Power == 1:
            Term = f"{Magnitude:.6g}x"
        else:
            Term = f"{Magnitude:.6g}x^{Power}"
        if not Pieces:
            Pieces.append(Term if Value >= 0 else f"-{Term}")
        else:
            Pieces.append(f" {Sign} {Term}")
    return "".join(Pieces) if Pieces else "0"


def _WriteCsv(Rows, CsvPath: Path) -> None:
    with CsvPath.open("w", newline="", encoding="utf-8") as Handle:
        Writer = Csv.DictWriter(Handle, fieldnames=list(Rows[0].keys()))
        Writer.writeheader()
        Writer.writerows(Rows)


def _MakePlot(SummaryRows, DpFit, GreedyFit, OutputPath: Path) -> None:
    XValues = Np.array([Row["InputSize"] for Row in SummaryRows], dtype=float)
    DpY = Np.array([Row["DpMedianMs"] for Row in SummaryRows], dtype=float)
    GreedyY = Np.array([Row["GreedyMedianMs"] for Row in SummaryRows], dtype=float)

    XDense = Np.linspace(float(Np.min(XValues)), float(Np.max(XValues)), 400)

    Plt.rcParams["font.family"] = "DejaVu Sans"
    Plt.style.use("seaborn-v0_8-whitegrid")
    Fig, Ax = Plt.subplots(figsize=(10, 6))
    Ax.scatter(XValues, DpY, color="#1f77b4", label="DP exacta", s=45)
    Ax.scatter(XValues, GreedyY, color="#d62728", label="Greedy", s=45)
    Ax.plot(XDense, DpFit["Polynomial"](XDense), color="#1f77b4", alpha=0.8)
    Ax.plot(XDense, GreedyFit["Polynomial"](XDense), color="#d62728", alpha=0.8)
    # Also draw lines connecting the measured median points (sorted by X)
    if len(XValues) > 1:
        Order = Np.argsort(XValues)
        Ax.plot(XValues[Order], DpY[Order], color="#1f77b4", linestyle='--', marker='o', alpha=0.7)
        Ax.plot(XValues[Order], GreedyY[Order], color="#d62728", linestyle='--', marker='o', alpha=0.7)
    Ax.set_xlabel("Tamaño de la entrada (L + n)")
    Ax.set_ylabel("Tiempo medio de ejecución (ms)")
    Ax.set_title("Cobertura de la valla: comparación de tiempos")
    Ax.legend()
    Fig.tight_layout()
    Fig.savefig(OutputPath, dpi=200)
    Plt.close(Fig)


def RunBenchmark(FenceLengths, Repetitions: int, Seed: int):
    Rows = []
    SummaryRows = []
    for FenceLength in FenceLengths:
        PainterCount = FenceLength * 2
        DpTimes = []
        GreedyTimes = []
        Ratios = []
        for Repetition in range(Repetitions):
            InstanceSeed = Seed + FenceLength * 1000 + Repetition
            RandomInstance = RandomModule.Random(InstanceSeed)
            Intervals = GenerateFeasibleInstance(FenceLength, PainterCount, RandomInstance)

            Exact, ExactMs = _TimeCall(SolveExactDp, Intervals, FenceLength)
            Greedy, GreedyMs = _TimeCall(SolveGreedyHeuristic, Intervals, FenceLength)

            if not Exact.Feasible:
                raise RuntimeError(f"Exact solver failed on FenceLength={FenceLength}")
            if not Greedy.Feasible:
                raise RuntimeError(f"Greedy solver failed on FenceLength={FenceLength}")

            Ratio = Greedy.TotalCost / Exact.TotalCost if Exact.TotalCost else float("nan")
            Rows.append(
                {
                    "FenceLength": FenceLength,
                    "PainterCount": PainterCount,
                    "InputSize": InstanceSize(FenceLength, PainterCount),
                    "Repetition": Repetition,
                    "ExactMs": ExactMs,
                    "GreedyMs": GreedyMs,
                    "OptimalCost": Exact.TotalCost,
                    "GreedyCost": Greedy.TotalCost,
                    "ApproximationRatio": Ratio,
                }
            )
            DpTimes.append(ExactMs)
            GreedyTimes.append(GreedyMs)
            Ratios.append(Ratio)

        SummaryRows.append(
            {
                "FenceLength": FenceLength,
                "PainterCount": PainterCount,
                "InputSize": InstanceSize(FenceLength, PainterCount),
                "DpMedianMs": float(Stats.median(DpTimes)) if DpTimes else 0.0,
                "GreedyMedianMs": float(Stats.median(GreedyTimes)) if GreedyTimes else 0.0,
                "RatioMean": float(Stats.mean(Ratios)) if Ratios else 1.0,
                "RatioMax": float(max(Ratios)) if Ratios else 1.0,
            }
        )

    return Rows, SummaryRows


def RunBenchmark_Subsets(SectionCounts, Repetitions: int, Seed: int):
    Rows = []
    SummaryRows = []
    for Sections in SectionCounts:
        PainterCount = max(Sections * 2, 4)
        DpTimes = []
        GreedyTimes = []
        Ratios = []
        Successful = 0
        # Require at least this many successful measurements per Sections
        MinSuccessful = min(3, Repetitions) if Repetitions > 0 else 3
        # Allow a limited number of attempts to reach MinSuccessful (avoid infinite loops)
        MaxAttempts = max(Repetitions * 3, MinSuccessful)
        Attempt = 0
        RepetitionIndex = 0
        while Successful < MinSuccessful and Attempt < MaxAttempts:
            InstanceSeed = Seed + Sections * 1000 + Attempt
            RandomInstance = RandomModule.Random(InstanceSeed)
            Painters = GenerateFeasibleInstance_Subsets(Sections, PainterCount, RandomInstance)

            Exact, ExactMs = _TimeCall(SolveExactDp_Subsets, Painters, Sections)
            Greedy, GreedyMs = _TimeCall(SolveGreedyHeuristic_Subsets, Painters, Sections)

            if not Exact.Feasible:
                print(f"Warning: Exact subset solver returned infeasible on Sections={Sections}, seed={InstanceSeed}; skipping attempt {Attempt}")
                Attempt += 1
                continue
            if not Greedy.Feasible:
                print(f"Warning: Greedy subset solver returned infeasible on Sections={Sections}, seed={InstanceSeed}; skipping attempt {Attempt}")
                Attempt += 1
                continue

            Ratio = Greedy.TotalCost / Exact.TotalCost if Exact.TotalCost else float("nan")
            Rows.append(
                {
                    "Sections": Sections,
                    "PainterCount": PainterCount,
                    "InputSize": Sections + PainterCount,
                    "Repetition": RepetitionIndex,
                    "ExactMs": ExactMs,
                    "GreedyMs": GreedyMs,
                    "OptimalCost": Exact.TotalCost,
                    "GreedyCost": Greedy.TotalCost,
                    "ApproximationRatio": Ratio,
                }
            )
            DpTimes.append(ExactMs)
            GreedyTimes.append(GreedyMs)
            Ratios.append(Ratio)
            Successful += 1
            RepetitionIndex += 1
            Attempt += 1

        if Successful < MinSuccessful:
            print(f"Warning: only collected {Successful} successful repetitions for Sections={Sections} (requested {MinSuccessful}), leaving NaN in summary")

        SummaryRows.append(
            {
                "Sections": Sections,
                "PainterCount": PainterCount,
                "InputSize": Sections + PainterCount,
                "SuccessfulReps": Successful,
                "DpMedianMs": float(Stats.median(DpTimes)) if DpTimes else float("nan"),
                "GreedyMedianMs": float(Stats.median(GreedyTimes)) if GreedyTimes else float("nan"),
                "RatioMean": float(Stats.mean(Ratios)) if Ratios else float("nan"),
                "RatioMax": float(max(Ratios)) if Ratios else float("nan"),
            }
        )

    return Rows, SummaryRows


def _WriteSummaryTex(SummaryRows, DpFit, GreedyFit, Counterexample, OutputPath: Path) -> None:
    CounterexampleInstance, Optimal, Greedy = Counterexample
    IntervalRows = []
    for IntervalItem in CounterexampleInstance:
        IntervalRows.append(
            f"{IntervalItem.Name or '-'} & {IntervalItem.Left} & {IntervalItem.Right} & {IntervalItem.Cost:.0f} \\\\")

    TableRows = []
    for Row in SummaryRows:
        TableRows.append(
            f"{Row['FenceLength']} & {Row['PainterCount']} & {Row['InputSize']} & "
            f"{Row['DpMedianMs']:.4f} & {Row['GreedyMedianMs']:.4f} & {Row['RatioMean']:.4f} \\\\")

    Content = rf"""% generado por scripts/Benchmark.py
\newcommand{{\DPPolynomial}}{{\ensuremath{{{_FormatPolynomial(DpFit['Coefficients'])}}}}}
\newcommand{{\GreedyPolynomial}}{{\ensuremath{{{_FormatPolynomial(GreedyFit['Coefficients'])}}}}}
\newcommand{{\DPDegree}}{{{DpFit['Degree']}}}
\newcommand{{\GreedyDegree}}{{{GreedyFit['Degree']}}}
\newcommand{{\DPRSquared}}{{{DpFit['RSquared']:.4f}}}
\newcommand{{\GreedyRSquared}}{{{GreedyFit['RSquared']:.4f}}}
\newcommand{{\DPAdjRSquared}}{{{DpFit['AdjRSquared']:.4f}}}
\newcommand{{\GreedyAdjRSquared}}{{{GreedyFit['AdjRSquared']:.4f}}}
\newcommand{{\DPRMSE}}{{{DpFit['Rmse']:.4f}}}
\newcommand{{\GreedyRMSE}}{{{GreedyFit['Rmse']:.4f}}}
\newcommand{{\MeanApproxRatio}}{{{float(Stats.mean([Row['RatioMean'] for Row in SummaryRows])):.4f}}}
\newcommand{{\WorstApproxRatio}}{{{float(max(Row['RatioMax'] for Row in SummaryRows)):.4f}}}

\begin{{tabular}}{{rrrrrr}}
\toprule
$L$ & $n$ & $L+n$ & DP med. (ms) & Greedy med. (ms) & Raz\'on media \\
\midrule
{chr(10).join(TableRows)}
\bottomrule
\end{{tabular}}

\paragraph{{Contraejemplo usado en la discusion de greedy choice property.}} La instancia encontrada por busqueda aleatoria contiene la siguiente lista de intervalos:

\begin{{tabular}}{{rrrr}}
\toprule
Nombre & Izquierda & Derecha & Costo \\
\midrule
{chr(10).join(IntervalRows)}
\bottomrule
\end{{tabular}}

La solucion optima exacta cuesta {Optimal.TotalCost:.0f} y la heuristica greedy cuesta {Greedy.TotalCost:.0f}.
"""
    OutputPath.write_text(Content, encoding="utf-8")


def _WriteSummaryTex_Subsets(SummaryRows, DpFit, GreedyFit, Counterexample, OutputPath: Path) -> None:
    # Counterexample: Painters list, show mask and cost
    Instance, Optimal, Greedy = Counterexample
    PainterRows = []
    for p in Instance:
        PainterRows.append(f"{p.Name or '-'} & {bin(p.Mask)} & {p.Cost:.0f} \\\\\\\\")

    TableRows = []
    for Row in SummaryRows:
        TableRows.append(
            
            f"{Row['Sections']} & {Row['PainterCount']} & {Row['InputSize']} & {Row['DpMedianMs']:.4f} & {Row['GreedyMedianMs']:.4f} & {Row['RatioMean']:.4f} \\\\\\\\")

    Content = rf"""% generado por scripts/Benchmark.py (subsets)
\newcommand{{\DPPolynomialSubsets}}{{\ensuremath{{{_FormatPolynomial(DpFit['Coefficients'])}}}}}
\newcommand{{\GreedyPolynomialSubsets}}{{\ensuremath{{{_FormatPolynomial(GreedyFit['Coefficients'])}}}}}
\newcommand{{\DPDegreeSubsets}}{{{DpFit['Degree']}}}
\newcommand{{\GreedyDegreeSubsets}}{{{GreedyFit['Degree']}}}
\newcommand{{\DPRSquaredSubsets}}{{{DpFit['RSquared']:.4f}}}
\newcommand{{\GreedyRSquaredSubsets}}{{{GreedyFit['RSquared']:.4f}}}

\begin{{tabular}}{{rrrrrr}}
    oprule
Sections & m & n & DP med. (ms) & Greedy med. (ms) & Raz\'on media \\
\midrule
{chr(10).join(TableRows)}
\bottomrule
\end{{tabular}}

\paragraph{{Contraejemplo (variante por subconjuntos).}} Instancia encontrada por busqueda aleatoria (máscara y costo):

\begin{{tabular}}{{rrr}}
    oprule
Nombre & Mascara & Costo \\
\midrule
{chr(10).join(PainterRows)}
\bottomrule
\end{{tabular}}

La solucion optima exacta cuesta {Optimal.TotalCost:.0f} y la heuristica greedy cuesta {Greedy.TotalCost:.0f}.
"""
    OutputPath.write_text(Content, encoding="utf-8")


def Main() -> None:
    Parser = Argparse.ArgumentParser(description="benchmark runner")
    Parser.add_argument("--seed", dest="Seed", type=int, default=2026)
    Parser.add_argument("--repetitions", dest="Repetitions", type=int, default=15)
    Parser.add_argument("--min-length", dest="MinLength", type=int, default=50)
    Parser.add_argument("--max-length", dest="MaxLength", type=int, default=1000)
    Parser.add_argument("--step", dest="Step", type=int, default=50)
    Args = Parser.parse_args()

    FenceLengths = list(range(Args.MinLength, Args.MaxLength + 1, Args.Step))
    Rows, SummaryRows = RunBenchmark(FenceLengths, Args.Repetitions, Args.Seed)

    CsvPath = ResultsDir / "Benchmark.csv"
    SummaryCsvPath = ResultsDir / "BenchmarkSummary.csv"
    PlotPath = ResultsDir / "BenchmarkScatter.png"
    SummaryTexPath = ResultsDir / "BenchmarkSummary.tex"
    CounterexampleJsonPath = ResultsDir / "Contraejemplo.json"

    _WriteCsv(Rows, CsvPath)
    _WriteCsv(SummaryRows, SummaryCsvPath)

    XValues = Np.array([Row["InputSize"] for Row in SummaryRows], dtype=float)
    DpY = Np.array([Row["DpMedianMs"] for Row in SummaryRows], dtype=float)
    GreedyY = Np.array([Row["GreedyMedianMs"] for Row in SummaryRows], dtype=float)
    DpFit = _FitBestPolynomial(XValues, DpY)
    GreedyFit = _FitBestPolynomial(XValues, GreedyY)

    Counterexample = FindGreedyCounterexample()
    Instance, Optimal, Greedy = Counterexample
    CounterexampleFenceLength = max(IntervalItem.Right for IntervalItem in Instance)
    CounterexampleJsonPath.write_text(
        Json.dumps(
            {
                "FenceLength": CounterexampleFenceLength,
                "Intervals": [
                    {
                        "Left": IntervalItem.Left,
                        "Right": IntervalItem.Right,
                        "Cost": IntervalItem.Cost,
                        "Name": IntervalItem.Name,
                    }
                    for IntervalItem in Instance
                ],
                "OptimalCost": Optimal.TotalCost,
                "GreedyCost": Greedy.TotalCost,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _MakePlot(SummaryRows, DpFit, GreedyFit, PlotPath)
    _WriteSummaryTex(SummaryRows, DpFit, GreedyFit, Counterexample, SummaryTexPath)

    # --- Run subset-variant benchmark (smaller scales) ---
    SectionCounts = list(range(4, 22, 2))
    SubRows, SubSummary = RunBenchmark_Subsets(SectionCounts, max(3, Args.Repetitions // 3), Args.Seed)

    SubCsvPath = ResultsDir / "BenchmarkSubsets.csv"
    SubSummaryCsvPath = ResultsDir / "BenchmarkSubsetsSummary.csv"
    SubPlotPath = ResultsDir / "BenchmarkSubsetsScatter.png"
    SubSummaryTexPath = ResultsDir / "BenchmarkSubsets.tex"
    SubCounterexamplePath = ResultsDir / "ContraejemploSubsets.json"

    _WriteCsv(SubRows, SubCsvPath)
    _WriteCsv(SubSummary, SubSummaryCsvPath)

    # Filter out summary rows with NaN medians (insufficient successful reps)
    ValidSubSummary = [Row for Row in SubSummary if not Np.isnan(Row.get("DpMedianMs", Np.nan)) and not Np.isnan(Row.get("GreedyMedianMs", Np.nan))]
    if not ValidSubSummary:
        print("Warning: no valid subset summary rows to fit/plot (all entries have insufficient successful repetitions)")
        # Provide empty default fits to avoid downstream crashes
        DpFit = {"Polynomial": (lambda x: Np.zeros_like(x)), "Coefficients": Np.array([0.0]), "Degree": 0, "RSquared": 0.0, "AdjRSquared": 0.0, "Rmse": 0.0, "BIC": 0.0}
        GreedyFit = DpFit
        _MakePlot(SubSummary, DpFit, GreedyFit, SubPlotPath)
    else:
        XValues = Np.array([Row["InputSize"] for Row in ValidSubSummary], dtype=float)
        DpY = Np.array([Row["DpMedianMs"] for Row in ValidSubSummary], dtype=float)
        GreedyY = Np.array([Row["GreedyMedianMs"] for Row in ValidSubSummary], dtype=float)
        DpFit = _FitBestPolynomial(XValues, DpY)
        GreedyFit = _FitBestPolynomial(XValues, GreedyY)

    # find a counterexample for subsets (small search)
    SubCounter = None
    Rng = RandomModule.Random(Args.Seed)
    for _ in range(2000):
        s = Rng.randint(4, 12)
        m = max(6, Rng.randint(6, 20))
        P = GenerateFeasibleInstance_Subsets(s, m, Rng)
        Exact = SolveExactDp_Subsets(P, s)
        Gre = SolveGreedyHeuristic_Subsets(P, s)
        if Exact.Feasible and Gre.Feasible and Gre.TotalCost > Exact.TotalCost:
            SubCounter = (P, Exact, Gre, s)
            break

    if SubCounter is None:
        # fallback trivial instance
        s = 4
        P = GenerateFeasibleInstance_Subsets(s, 8, RandomModule.Random(1))
        Exact = SolveExactDp_Subsets(P, s)
        Gre = SolveGreedyHeuristic_Subsets(P, s)
        SubCounter = (P, Exact, Gre, s)

    _MakePlot(SubSummary, DpFit, GreedyFit, SubPlotPath)
    _WriteSummaryTex_Subsets(SubSummary, DpFit, GreedyFit, SubCounter[:3], SubSummaryTexPath)

    P, Exact, Gre, s = SubCounter
    SubCounterexamplePath.write_text(Json.dumps({
        "Sections": s,
        "Painters": [ {"Mask": p.Mask, "Cost": p.Cost, "Name": p.Name} for p in P ],
        "OptimalCost": Exact.TotalCost,
        "GreedyCost": Gre.TotalCost,
    }, indent=2), encoding="utf-8")

    print(f"Wrote {SubCsvPath}")
    print(f"Wrote {SubSummaryCsvPath}")
    print(f"Wrote {SubPlotPath}")
    print(f"Wrote {SubSummaryTexPath}")
    print(f"Wrote {SubCounterexamplePath}")

    print(f"Wrote {CsvPath}")
    print(f"Wrote {SummaryCsvPath}")
    print(f"Wrote {PlotPath}")
    print(f"Wrote {SummaryTexPath}")
    print(f"Wrote {CounterexampleJsonPath}")
    print(f"DP fit:     degree {DpFit['Degree']}, R²={DpFit['RSquared']:.4f}, R²_adj={DpFit['AdjRSquared']:.4f}, BIC={DpFit['BIC']:.2f}")
    print(f"Greedy fit: degree {GreedyFit['Degree']}, R²={GreedyFit['RSquared']:.4f}, R²_adj={GreedyFit['AdjRSquared']:.4f}, BIC={GreedyFit['BIC']:.2f}")


if __name__ == "__main__":
    Main()




