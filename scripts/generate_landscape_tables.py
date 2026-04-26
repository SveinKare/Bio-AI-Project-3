#!/usr/bin/env python3
"""
Build paper-style landscape summary tables (NSGA-II, PSO, optionally SGA) from
repeated-run CSV produced by run_stats.sh (output/stats/runs_raw.csv).

Columns match the single-objective GA table format:
  Landscape | Average accuracy | Std deviation | Best found accuracy | Best found gene | Global optimum

Global optima are read from output/best/landscape_global_optima.txt (same source as run.sh).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

CASE_ORDER = ["triangle", "breast", "credit", "letter"]
CASE_DISPLAY = {
    "triangle": "Triangle",
    "breast": "Breast",
    "credit": "Credit",
    "letter": "Letter",
}

ALGO_LABEL = {
    "nsga2": "NSGA-II",
    "pso": "PSO",
    "sga": "single-objective GA",
}


def parse_global_optima(path: Path) -> dict[str, float]:
    """Map case key -> optimum accuracy (fitness)."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, float] = {}
    # Section title -> case key
    blocks = [
        ("triangle", r"Triangle:"),
        ("breast", r"Breast Cancer:"),
        ("credit", r"Credit:"),
        ("letter", r"Letter:"),
    ]
    for case, header in blocks:
        m = re.search(
            rf"{header}\s*.*?Fitness:\s*([\d.+-]+)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not m:
            print(f"Warning: could not parse global optimum for {case}", file=sys.stderr)
            continue
        out[case] = float(m.group(1))
    return out


def fmt_acc(x: float) -> str:
    """Display accuracies / optima (six decimals; integer-like optima as n.0)."""
    if pd.isna(x):
        return ""
    xf = float(x)
    if abs(xf - 4.0) < 1e-9:
        return "4.0"
    return f"{xf:.6f}"


def fmt_std(x: float) -> str:
    """Std deviation: always six decimals (zeros show as 0.000000)."""
    if pd.isna(x):
        return ""
    return f"{float(x):.6f}"


def build_table(
    df: pd.DataFrame,
    algorithm: str,
    optima: dict[str, float],
) -> pd.DataFrame:
    sub = df[df["algorithm"] == algorithm].copy()
    if sub.empty:
        raise ValueError(f"No rows for algorithm={algorithm!r}")

    rows: list[dict[str, object]] = []
    for case in CASE_ORDER:
        g = sub[sub["case"] == case]
        if g.empty:
            rows.append(
                {
                    "Landscape": CASE_DISPLAY[case],
                    "Average accuracy": "",
                    "Std deviation": "",
                    "Best found accuracy": "",
                    "Best found gene": "",
                    "Global optimum": fmt_acc(optima.get(case, float("nan"))),
                }
            )
            continue

        mean_acc = float(g["accuracy"].mean())
        std_acc = float(g["accuracy"].std(ddof=1))
        if pd.isna(std_acc):
            std_acc = 0.0

        idx = g["accuracy"].idxmax()
        best_acc = float(g.loc[idx, "accuracy"])
        best_gene = str(g.loc[idx, "bitstring"])

        rows.append(
            {
                "Landscape": CASE_DISPLAY[case],
                "Average accuracy": mean_acc,
                "Std deviation": std_acc,
                "Best found accuracy": best_acc,
                "Best found gene": best_gene,
                "Global optimum": optima.get(case, float("nan")),
            }
        )

    return pd.DataFrame(rows)


def _format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """String copy with fixed decimals for numeric columns (paper-style)."""
    disp = df.copy()
    for col in ("Average accuracy", "Best found accuracy", "Global optimum"):
        if col not in disp.columns:
            continue
        disp[col] = disp[col].map(
            lambda x: fmt_acc(float(x))
            if isinstance(x, (int, float)) and not pd.isna(x)
            else ("" if pd.isna(x) else x)
        )
    if "Std deviation" in disp.columns:
        disp["Std deviation"] = disp["Std deviation"].map(
            lambda x: fmt_std(float(x))
            if isinstance(x, (int, float)) and not pd.isna(x)
            else ("" if pd.isna(x) else x)
        )
    return disp


def markdown_pipe_table(df: pd.DataFrame) -> str:
    """GitHub-style pipe table (no extra dependency)."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def df_to_markdown(df: pd.DataFrame) -> str:
    disp = _format_display_df(df)
    return markdown_pipe_table(disp)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate landscape summary tables (NSGA-II, PSO, SGA) from runs_raw.csv"
    )
    p.add_argument(
        "runs_csv",
        nargs="?",
        default="output/stats/runs_raw.csv",
        help="CSV from run_stats.sh (default: output/stats/runs_raw.csv)",
    )
    p.add_argument(
        "--global-optima",
        type=Path,
        default=Path("output/best/landscape_global_optima.txt"),
        help="Global optima reference file",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/stats"),
        help="Directory for table CSV/Markdown files",
    )
    p.add_argument(
        "--algorithms",
        default="nsga2,pso,sga",
        help="Comma-separated: nsga2, pso, sga (default: all three)",
    )
    args = p.parse_args()

    runs_path = Path(args.runs_csv)
    if not runs_path.is_file():
        print(
            f"Missing {runs_path}. Run ./run_stats.sh to generate repeated runs.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(runs_path, dtype={"bitstring": str})
    for col in ("algorithm", "case", "accuracy", "bitstring"):
        if col not in df.columns:
            print(f"Missing column {col!r} in {runs_path}", file=sys.stderr)
            return 1

    if not args.global_optima.is_file():
        print(f"Missing global optima file {args.global_optima}", file=sys.stderr)
        return 1
    optima = parse_global_optima(args.global_optima)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    for algo in algos:
        label = ALGO_LABEL.get(algo, algo)
        table = build_table(df, algo, optima)
        stem = f"table_{algo}_landscapes"
        csv_path = args.out_dir / f"{stem}.csv"
        md_path = args.out_dir / f"{stem}.md"

        # Round floats for stable CSV (avoid 1e-16 noise)
        csv_out = table.copy()
        for col in (
            "Average accuracy",
            "Std deviation",
            "Best found accuracy",
            "Global optimum",
        ):
            if col in csv_out.columns:
                csv_out[col] = csv_out[col].map(
                    lambda x: round(float(x), 10) if isinstance(x, (int, float)) and not pd.isna(x) else x
                )
        csv_out.to_csv(csv_path, index=False)
        caption = (
            f"Results of **{label}** on the four training landscapes. "
            "Each row aggregates all runs in `runs_raw.csv` for that landscape "
            "(mean and sample standard deviation of **accuracy** per run; "
            "best accuracy and corresponding bitstring across runs; global optimum for reference). "
            "Bit-strings are in **big-endian** order (same as `main` output)."
        )
        md_path.write_text(
            caption + "\n\n" + df_to_markdown(table) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")

    combined = args.out_dir / f"tables_{'_'.join(algos)}_landscapes.md"
    title_algos = ", ".join(ALGO_LABEL.get(a, a) for a in algos)
    parts = [
        f"# Landscape summary tables ({title_algos})\n",
        "\nData source: `{}` · Global optima: `{}`\n".format(
            runs_path.as_posix(), args.global_optima.as_posix()
        ),
    ]
    for algo in algos:
        label = ALGO_LABEL.get(algo, algo)
        table = build_table(df, algo, optima)
        parts.append(f"\n## {label}\n\n")
        parts.append(
            "Each row aggregates all runs for that landscape: **average accuracy** and "
            "**std. dev.** across runs, **best found accuracy** and **gene** at that best run, "
            "and **global optimum** for reference.\n\n"
        )
        parts.append(df_to_markdown(table))
        parts.append("\n")
    combined.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {combined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
