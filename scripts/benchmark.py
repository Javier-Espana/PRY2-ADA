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


def _FormatSignedFloat(Value: float) -> str:
    return f"{Value:.6g}" if Value >= 0 else f"-{abs(Value):.6g}"


def _EvaluateFit(XValues: Np.ndarray, YValues: Np.ndarray, Prediction, *, Name: str, DisplayName: str, Equation: str, Parameters: int, Degree: int | None = None):
    Predicted = Prediction(XValues)
    Residuals = YValues - Predicted
    Mse = float(Np.mean(Residuals**2)) if len(YValues) else 0.0
    Rmse = float(Np.sqrt(Mse))
    SsRes = float(Np.sum(Residuals**2))
    SsTot = float(Np.sum((YValues - Np.mean(YValues)) ** 2))
    RSquared = 1.0 if SsTot == 0 else 1.0 - SsRes / SsTot
    N = len(XValues)
    AdjRSquared = (
        1.0 - (1.0 - RSquared) * (N - 1) / max(1, N - Parameters)
        if SsTot != 0 else 1.0
    )
    Bic = N * Np.log(max(Mse, 1e-300)) + Parameters * Np.log(max(N, 2)) if Mse > 0 else -float("inf")
    return {
        "Name": Name,
        "DisplayName": DisplayName,
        "Equation": Equation,
        "Predict": Prediction,
        "Parameters": Parameters,
        "Degree": Degree,
        "Mse": Mse,
        "Rmse": Rmse,
        "RSquared": RSquared,
        "AdjRSquared": AdjRSquared,
        "BIC": Bic,
    }


def _FitPolynomialModel(XValues: Np.ndarray, YValues: Np.ndarray, Degree: int, DisplayName: str | None = None):
    Coefficients = Np.polyfit(XValues, YValues, Degree)
    Polynomial = Np.poly1d(Coefficients)
    Equation = _FormatPolynomial(Coefficients)
    DisplayName = DisplayName or f"polinomial grado {Degree}"
    return _EvaluateFit(
        XValues,
        YValues,
        Polynomial,
        Name=f"poly{Degree}",
        DisplayName=DisplayName,
        Equation=Equation,
        Parameters=Degree + 1,
        Degree=Degree,
    )


def _FitExponentialModel(XValues: Np.ndarray, YValues: Np.ndarray):
    if Np.any(YValues <= 0):
        return None
    Slope, Intercept = Np.polyfit(XValues, Np.log(YValues), 1)
    Scale = float(Np.exp(Intercept))

    def Predict(Values):
        return Scale * Np.exp(Slope * Values)

    Equation = f"{Scale:.6g}e^{{{Slope:.6g}x}}"
    return _EvaluateFit(
        XValues,
        YValues,
        Predict,
        Name="exp",
        DisplayName="exponencial",
        Equation=Equation,
        Parameters=2,
    )


def _FitPowerModel(XValues: Np.ndarray, YValues: Np.ndarray):
    if Np.any(XValues <= 0) or Np.any(YValues <= 0):
        return None
    Slope, Intercept = Np.polyfit(Np.log(XValues), Np.log(YValues), 1)
    Scale = float(Np.exp(Intercept))

    def Predict(Values):
        return Scale * (Values ** Slope)

    Equation = f"{Scale:.6g}x^{{{Slope:.6g}}}"
    return _EvaluateFit(
        XValues,
        YValues,
        Predict,
        Name="power",
        DisplayName="potencia",
        Equation=Equation,
        Parameters=2,
    )


def _FitNLogNModel(XValues: Np.ndarray, YValues: Np.ndarray):
    if Np.any(XValues <= 1):
        return None
    Basis = XValues * Np.log(XValues)
    Design = Np.column_stack([Np.ones_like(Basis), Basis])
    Coefficients, *_ = Np.linalg.lstsq(Design, YValues, rcond=None)
    Intercept = float(Coefficients[0])
    Slope = float(Coefficients[1])

    def Predict(Values):
        return Intercept + Slope * (Values * Np.log(Values))

    Sign = " + " if Slope >= 0 else " - "
    Equation = f"{Intercept:.6g}{Sign}{abs(Slope):.6g}x\\log x"
    return _EvaluateFit(
        XValues,
        YValues,
        Predict,
        Name="nlogn",
        DisplayName="n log n",
        Equation=Equation,
        Parameters=2,
    )


def _SelectBestFit(XValues: Np.ndarray, YValues: Np.ndarray, *, Algorithm: str):
    if len(XValues) < 2:
        Value = float(YValues[0]) if len(YValues) else 0.0

        def Predict(Values):
            return Np.full_like(Values, Value, dtype=float)

        return {
            "Name": "const",
            "DisplayName": f"constante ({Algorithm})",
            "Equation": f"{Value:.6g}",
            "Predict": Predict,
            "Parameters": 1,
            "Degree": 0,
            "Mse": 0.0,
            "Rmse": 0.0,
            "RSquared": 1.0,
            "AdjRSquared": 1.0,
            "BIC": -float("inf"),
        }

    if Algorithm == "DP":
        Candidates = [
            candidate for candidate in [
                _FitExponentialModel(XValues, YValues),
                _FitPowerModel(XValues, YValues),
            ] if candidate is not None
        ]
    elif Algorithm == "Greedy":
        Candidates = [
            candidate for candidate in [
                _FitPolynomialModel(XValues, YValues, 2, DisplayName="cuadrático"),
                _FitNLogNModel(XValues, YValues),
            ] if candidate is not None
        ]
    else:
        Candidates = []

    if not Candidates:
        MaxPolynomialDegree = min(4, len(XValues) - 1)
        for Degree in range(1, MaxPolynomialDegree + 1):
            Candidates.append(_FitPolynomialModel(XValues, YValues, Degree))

    if not Candidates:
        raise RuntimeError(f"no fit candidates could be computed for {Algorithm}")

    return min(Candidates, key=lambda Item: Item["BIC"])


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
    Ax.plot(XDense, DpFit["Predict"](XDense), color="#1f77b4", alpha=0.9, linewidth=2.2, label=f"DP ajuste ({DpFit['DisplayName']})")
    Ax.plot(XDense, GreedyFit["Predict"](XDense), color="#d62728", alpha=0.9, linewidth=2.2, label=f"Greedy ajuste ({GreedyFit['DisplayName']})")
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
\newcommand{{\DPFitNameSubsets}}{{{DpFit['DisplayName']}}}
\newcommand{{\GreedyFitNameSubsets}}{{{GreedyFit['DisplayName']}}}
\newcommand{{\DPFitEquationSubsets}}{{\ensuremath{{{DpFit['Equation']}}}}}
\newcommand{{\GreedyFitEquationSubsets}}{{\ensuremath{{{GreedyFit['Equation']}}}}}
\newcommand{{\DPFitBICSubsets}}{{{DpFit['BIC']:.4f}}}
\newcommand{{\GreedyFitBICSubsets}}{{{GreedyFit['BIC']:.4f}}}
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
    Parser.add_argument("--min-sections", dest="MinSections", type=int, default=4)
    Parser.add_argument("--max-sections", dest="MaxSections", type=int, default=18)
    Parser.add_argument("--step", dest="Step", type=int, default=1)
    Args = Parser.parse_args()

    SectionCounts = list(range(Args.MinSections, Args.MaxSections + 1, Args.Step))
    Rows, SummaryRows = RunBenchmark_Subsets(SectionCounts, Args.Repetitions, Args.Seed)

    CsvPath = ResultsDir / "BenchmarkSubsets.csv"
    SummaryCsvPath = ResultsDir / "BenchmarkSubsetsSummary.csv"
    PlotPath = ResultsDir / "BenchmarkSubsetsScatter.png"
    SummaryTexPath = ResultsDir / "BenchmarkSubsets.tex"
    CounterexampleJsonPath = ResultsDir / "ContraejemploSubsets.json"

    _WriteCsv(Rows, CsvPath)
    _WriteCsv(SummaryRows, SummaryCsvPath)

    # Filter out summary rows with NaN medians (insufficient successful reps)
    ValidSummary = [Row for Row in SummaryRows if not Np.isnan(Row.get("DpMedianMs", Np.nan)) and not Np.isnan(Row.get("GreedyMedianMs", Np.nan))]
    if not ValidSummary:
        print("Warning: no valid subset summary rows to fit/plot (all entries have insufficient successful repetitions)")
        # Provide empty default fits to avoid downstream crashes
        DpFit = {"Predict": (lambda x: Np.zeros_like(x)), "DisplayName": "constante", "Equation": "0", "Degree": 0, "RSquared": 0.0, "AdjRSquared": 0.0, "Rmse": 0.0, "BIC": 0.0}
        GreedyFit = DpFit
        _MakePlot(SummaryRows, DpFit, GreedyFit, PlotPath)
    else:
        XValues = Np.array([Row["InputSize"] for Row in ValidSummary], dtype=float)
        DpY = Np.array([Row["DpMedianMs"] for Row in ValidSummary], dtype=float)
        GreedyY = Np.array([Row["GreedyMedianMs"] for Row in ValidSummary], dtype=float)
        DpFit = _SelectBestFit(XValues, DpY, Algorithm="DP")
        GreedyFit = _SelectBestFit(XValues, GreedyY, Algorithm="Greedy")

    # find a counterexample for subsets (small search)
    Counterexample = None
    Rng = RandomModule.Random(Args.Seed)
    for _ in range(2000):
        s = Rng.randint(4, 12)
        m = max(6, Rng.randint(6, 20))
        P = GenerateFeasibleInstance_Subsets(s, m, Rng)
        Exact = SolveExactDp_Subsets(P, s)
        Gre = SolveGreedyHeuristic_Subsets(P, s)
        if Exact.Feasible and Gre.Feasible and Gre.TotalCost > Exact.TotalCost:
            Counterexample = (P, Exact, Gre, s)
            break

    if Counterexample is None:
        # fallback trivial instance
        s = 4
        P = GenerateFeasibleInstance_Subsets(s, 8, RandomModule.Random(1))
        Exact = SolveExactDp_Subsets(P, s)
        Gre = SolveGreedyHeuristic_Subsets(P, s)
        Counterexample = (P, Exact, Gre, s)

    _MakePlot(SummaryRows, DpFit, GreedyFit, PlotPath)
    _WriteSummaryTex_Subsets(SummaryRows, DpFit, GreedyFit, Counterexample[:3], SummaryTexPath)

    P, Exact, Gre, s = Counterexample
    CounterexampleJsonPath.write_text(Json.dumps({
        "Sections": s,
        "Painters": [ {"Mask": p.Mask, "Cost": p.Cost, "Name": p.Name} for p in P ],
        "OptimalCost": Exact.TotalCost,
        "GreedyCost": Gre.TotalCost,
    }, indent=2), encoding="utf-8")

    print(f"Wrote {CsvPath}")
    print(f"Wrote {SummaryCsvPath}")
    print(f"Wrote {PlotPath}")
    print(f"Wrote {SummaryTexPath}")
    print(f"Wrote {CounterexampleJsonPath}")
    print(f"DP fit:     {DpFit['DisplayName']} -> {DpFit['Equation']} | R²={DpFit['RSquared']:.4f}, R²_adj={DpFit['AdjRSquared']:.4f}, BIC={DpFit['BIC']:.2f}")
    print(f"Greedy fit: {GreedyFit['DisplayName']} -> {GreedyFit['Equation']} | R²={GreedyFit['RSquared']:.4f}, R²_adj={GreedyFit['AdjRSquared']:.4f}, BIC={GreedyFit['BIC']:.2f}")


if __name__ == "__main__":
    Main()




