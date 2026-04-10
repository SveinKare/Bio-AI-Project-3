#!/usr/bin/env python3
"""
Visualise Binary PSO results: swarm convergence, fitness, accuracy, and diversity.

Usage:
    python plot_pso.py --prefix output/pso-breast-w --save figures
    python plot_pso.py --prefix output/pso-breast-w output/pso-credit-a output/pso-letter-r output/pso-triangle --save figures
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="viridis")


def load_csv(prefix: str, suffix: str) -> pd.DataFrame | None:
    path = f"{prefix}_{suffix}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── Plot 1: Swarm in objective space, coloured by iteration ──────────────────

def plot_swarm_objective_space(pareto: pd.DataFrame, pop: pd.DataFrame,
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
        cbar.set_label("Iteration", fontsize=9)
    elif pop is not None:
        ax.scatter(pop["time"], pop["accuracy"],
                   s=10, alpha=0.3, color="steelblue", label="swarm",
                   rasterized=True)

    if pareto is not None and not pareto.empty:
        ax.scatter(pareto["time"], pareto["accuracy"],
                   s=120, color="gold", edgecolors="black", linewidths=1.2,
                   zorder=6, marker="*", label=f"Global best")

    ax.set_xlabel("Time / complexity")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Swarm Exploration -- {stem}")
    ax.legend(fontsize="x-small")


# ── Plot 2: Fitness convergence (gbest + mean) ───────────────────────────────

def plot_fitness_convergence(gens: pd.DataFrame, ax: plt.Axes, stem: str):
    ax.plot(gens["generation"], gens["hypervolume"],
            "r-", linewidth=2.5, label="global best fitness")
    ax.plot(gens["generation"], gens["best_accuracy"],
            "g--", linewidth=1.5, alpha=0.8, label="best accuracy (raw)")
    ax.plot(gens["generation"], gens["mean_accuracy"],
            "b:", linewidth=1.5, alpha=0.6, label="mean accuracy (raw)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title(f"Convergence -- {stem}")
    ax.legend(fontsize="x-small", loc="lower right")


# ── Plot 3: Accuracy vs features for final swarm ─────────────────────────────

def plot_accuracy_vs_features(pop: pd.DataFrame, pareto: pd.DataFrame,
                               ax: plt.Axes, stem: str):
    if pop is not None and not pop.empty:
        ax.scatter(pop["num_features"], pop["accuracy"],
                   s=20, alpha=0.4, color="steelblue", label="final swarm",
                   rasterized=True)

    if pareto is not None and not pareto.empty:
        ax.scatter(pareto["num_features"], pareto["accuracy"],
                   s=120, color="gold", edgecolors="black", linewidths=1.2,
                   zorder=6, marker="*",
                   label=f"best ({int(pareto.iloc[0]['num_features'])} feat)")

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Features -- {stem}")
    ax.legend(fontsize="x-small")


# ── Plot 4: Diversity & inertia weight ────────────────────────────────────────

def plot_diversity_mutation(gens: pd.DataFrame, ax: plt.Axes, stem: str):
    ax.plot(gens["generation"], gens["diversity"],
            "m-", linewidth=2, label="diversity (unique/swarm)")
    ax.set_ylabel("Diversity")

    ax2 = ax.twinx()
    ax2.plot(gens["generation"], gens["mutation_rate"],
             "r--", linewidth=1.5, alpha=0.8, label="mutation rate")
    ax2.set_ylabel("Mutation rate", fontsize=12)

    ax.set_xlabel("Iteration")
    ax.set_title(f"Diversity & Mutation -- {stem}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize="x-small", loc="center right")


# ── Per-prefix driver ────────────────────────────────────────────────────────

def visualise_pso(prefix: str, save_dir: str | None):
    stem = os.path.basename(prefix)

    pareto = load_csv(prefix, "pareto")
    gens = load_csv(prefix, "gens")
    pop = load_csv(prefix, "pop")
    snapshots = load_csv(prefix, "snapshots")

    if gens is None:
        sys.exit(f"File not found: {prefix}_gens.csv")

    gbest_fit = gens["hypervolume"].iloc[-1] if not gens.empty else "?"
    print(f"  {stem}: iterations={len(gens)}, gbest_fitness={gbest_fit}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Binary PSO Results -- {stem}",
                 fontsize=16, fontweight="bold")

    plot_swarm_objective_space(pareto, pop, axes[0, 0], stem, snapshots)
    plot_fitness_convergence(gens, axes[0, 1], stem)
    plot_accuracy_vs_features(pop, pareto, axes[1, 0], stem)
    plot_diversity_mutation(gens, axes[1, 1], stem)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"pso_{stem}.png")
        fig.savefig(out, dpi=150)
        print(f"    Saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise Binary PSO results")
    parser.add_argument("--prefix", nargs="+", required=True,
                        help="Output prefix(es) from PSO run "
                             "(e.g. output/pso-breast-w)")
    parser.add_argument("--save", default=None,
                        help="Directory to save PNGs instead of showing")
    args = parser.parse_args()

    for prefix in args.prefix:
        print(f"Processing {prefix} ...")
        visualise_pso(prefix, args.save)


if __name__ == "__main__":
    main()
