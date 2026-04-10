#!/usr/bin/env python3
"""
Visualise NSGA-II results: Pareto front, convergence, and population diversity.

Usage:
    python plot_nsga2.py --prefix output/breast-w --save figures
    python plot_nsga2.py --prefix output/breast-w output/credit-a output/letter-r output/triangle --save figures
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="viridis")


def load_pareto(prefix: str) -> pd.DataFrame:
    path = f"{prefix}_pareto.csv"
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    return pd.read_csv(path)


def load_gens(prefix: str) -> pd.DataFrame:
    path = f"{prefix}_gens.csv"
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    return pd.read_csv(path)


def load_pop(prefix: str) -> pd.DataFrame:
    path = f"{prefix}_pop.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_snapshots(prefix: str) -> pd.DataFrame | None:
    path = f"{prefix}_snapshots.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Plot 1: Pareto front with per-generation colour coding ───────────────────

def plot_pareto_front(pareto: pd.DataFrame, pop: pd.DataFrame,
                      ax: plt.Axes, stem: str,
                      snapshots: pd.DataFrame | None = None):
    if snapshots is not None and not snapshots.empty:
        max_gen = snapshots["generation"].max()
        sc = ax.scatter(
            snapshots["time"], snapshots["accuracy"],
            c=snapshots["generation"], cmap="plasma",
            s=6, alpha=0.25, vmin=0, vmax=max_gen,
            rasterized=True, zorder=2)
        cbar = ax.figure.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label("Generation", fontsize=9)
    elif pop is not None:
        dominated = pop[pop["rank"] > 0]
        ax.scatter(dominated["time"], dominated["accuracy"],
                   s=10, alpha=0.3, color="steelblue", label="dominated",
                   rasterized=True)

    ax.scatter(pareto["time"], pareto["accuracy"],
               s=40, color="red", edgecolors="darkred", linewidths=0.5,
               zorder=5, label=f"Pareto front ({len(pareto)})")

    sorted_p = pareto.sort_values("time")
    ax.step(sorted_p["time"], sorted_p["accuracy"],
            where="post", color="red", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Training time (minimize)")
    ax.set_ylabel("Accuracy (maximize)")
    ax.set_title(f"Pareto Front -- {stem}")
    ax.legend(fontsize="x-small")


# ── Plot 2: Convergence (hypervolume + best accuracy) ───────────────────────

def plot_convergence(gens: pd.DataFrame, ax: plt.Axes, stem: str):
    ax.plot(gens["generation"], gens["best_accuracy"],
            "r-", linewidth=2, label="best accuracy")
    ax.plot(gens["generation"], gens["mean_accuracy"],
            "b--", linewidth=1.5, alpha=0.7, label="mean accuracy")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Convergence -- {stem}")
    ax.legend(fontsize="x-small", loc="lower right")


# ── Plot 3: Hypervolume over generations ─────────────────────────────────────

def plot_hypervolume(gens: pd.DataFrame, ax: plt.Axes, stem: str):
    ax.plot(gens["generation"], gens["hypervolume"],
            "g-", linewidth=2, label="hypervolume")
    ax.fill_between(gens["generation"], 0, gens["hypervolume"],
                    alpha=0.15, color="green")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Hypervolume -- {stem}")
    ax.legend(fontsize="x-small")


# ── Plot 4: Adaptive mutation & diversity ─────────────────────────────────────

def plot_diversity(gens: pd.DataFrame, ax: plt.Axes, stem: str):
    has_diversity = "diversity" in gens.columns
    has_mutation = "mutation_rate" in gens.columns

    if has_diversity:
        ax.plot(gens["generation"], gens["diversity"],
                "m-", linewidth=2, label="diversity (unique/pop)")
        ax.set_ylabel("Diversity")
    else:
        ax.plot(gens["generation"], gens["pareto_size"],
                "m-", linewidth=2, label="Pareto front size")
        ax.set_ylabel("Pareto front size")

    ax2 = ax.twinx()
    if has_mutation:
        ax2.plot(gens["generation"], gens["mutation_rate"],
                 "r--", linewidth=1.5, alpha=0.8, label="mutation rate")
        ax2.set_ylabel("Mutation rate", fontsize=12)
    else:
        ax2.plot(gens["generation"], gens["mean_time"],
                 "c--", linewidth=1.5, alpha=0.7, label="mean time")
        ax2.set_ylabel("Mean time", fontsize=12)

    ax.set_xlabel("Generation")
    ax.set_title(f"Adaptive Diversity -- {stem}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize="x-small", loc="center right")


# ── Per-prefix driver ────────────────────────────────────────────────────────

def visualise_nsga2(prefix: str, save_dir: str | None):
    stem = os.path.basename(prefix)
    pareto = load_pareto(prefix)
    gens = load_gens(prefix)
    pop = load_pop(prefix)
    snapshots = load_snapshots(prefix)

    print(f"  {stem}: Pareto front={len(pareto)}, generations={len(gens)}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"NSGA-II Results -- {stem}",
                 fontsize=16, fontweight="bold")

    plot_pareto_front(pareto, pop, axes[0, 0], stem, snapshots)
    plot_convergence(gens, axes[0, 1], stem)
    plot_hypervolume(gens, axes[1, 0], stem)
    plot_diversity(gens, axes[1, 1], stem)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"nsga2_{stem}.png")
        fig.savefig(out, dpi=150)
        print(f"    Saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise NSGA-II results")
    parser.add_argument("--prefix", nargs="+", required=True,
                        help="Output prefix(es) from NSGA-II run "
                             "(e.g. output/breast-w)")
    parser.add_argument("--save", default=None,
                        help="Directory to save PNGs instead of showing")
    args = parser.parse_args()

    for prefix in args.prefix:
        print(f"Processing {prefix} ...")
        visualise_nsga2(prefix, args.save)


if __name__ == "__main__":
    main()
