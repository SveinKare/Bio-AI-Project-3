#!/usr/bin/env python3
"""Aggregate repeated algorithm runs: mean / std / count of scalar_fitness per group."""

import argparse
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Summarize stats runs (mean, std of scalar_fitness).")
    p.add_argument("input_csv", help="Path to runs_raw.csv from run_stats.sh")
    p.add_argument("--out-csv", default=None, help="Write summary table CSV path")
    p.add_argument("--out-md", default=None, help="Optional Markdown table path")
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    if "scalar_fitness" not in df.columns:
        print("Missing column scalar_fitness", file=sys.stderr)
        sys.exit(1)

    g = (
        df.groupby(["algorithm", "case"], as_index=False)
        .agg(
            n_runs=("scalar_fitness", "count"),
            mean_scalar_fitness=("scalar_fitness", "mean"),
            std_scalar_fitness=("scalar_fitness", "std"),
            mean_accuracy=("accuracy", "mean"),
        )
    )
    g["std_scalar_fitness"] = g["std_scalar_fitness"].fillna(0.0)

    algo_order = ["nsga2", "pso", "sga"]
    case_order = ["breast", "credit", "letter", "triangle"]
    g["algorithm"] = pd.Categorical(g["algorithm"], algo_order, ordered=True)
    g["case"] = pd.Categorical(g["case"], case_order, ordered=True)
    g = g.sort_values(["case", "algorithm"]).reset_index(drop=True)

    if args.out_csv:
        g.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")

    print()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()

    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write("# Statistical summary (scalar fitness)\n\n")
            f.write(
                "Comparison uses **`scalar_fitness`**: accuracy minus feature-count penalty "
                "(`(k/N)*epsilon`) for NSGA-II best-by-accuracy individual; PSO global-best "
                "`fitness`; SGA best individual `accuracy - penalty`.\n\n"
            )
            try:
                f.write(g.to_markdown(index=False, floatfmt=".6f"))
            except ImportError:
                f.write("```\n")
                f.write(g.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
                f.write("\n```\n")
            f.write("\n")
        print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
