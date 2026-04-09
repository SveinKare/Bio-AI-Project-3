#!/usr/bin/env python3
"""
Visualise a feature-selection fitness landscape exported as CSV,
with every local optimum explicitly highlighted.

Usage:
    python plot_landscape.py <landscape.csv> [--save <output_dir>]
    python plot_landscape.py data/*.csv --save figures

The CSV is produced by:  ./build/main <file.h5> --export landscape.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3D projection
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid", context="talk", palette="viridis")

# ── Local-optima detection ───────────────────────────────────────────────────

def find_local_optima(df: pd.DataFrame, n_features: int,
                      include_plateaus: bool = False) -> np.ndarray:
    """Return a boolean mask over *df* rows that are local optima.

    A strict local optimum (peak) has fitness >= every Hamming-1 neighbour
    (with strict > for at least one neighbour).
    If include_plateaus is True, a point where *all* neighbours have equal
    fitness is also counted as an optimum.
    """
    fitness = dict(zip(df["index"].values, df["mean_accuracy"].values))
    max_index = (1 << n_features) - 1
    is_optimum = np.zeros(len(df), dtype=bool)

    for row_pos, (idx, fit) in enumerate(
            zip(df["index"].values, df["mean_accuracy"].values)):
        dominated = False
        strictly_better = False
        for bit in range(n_features):
            nb = idx ^ (1 << bit)
            if nb < 1 or nb > max_index:
                continue
            nb_fit = fitness.get(nb)
            if nb_fit is None:
                continue
            if nb_fit > fit:
                dominated = True
                break
            if fit > nb_fit:
                strictly_better = True

        if dominated:
            continue

        if strictly_better:
            is_optimum[row_pos] = True
        elif include_plateaus:
            is_optimum[row_pos] = True

    return is_optimum


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ── Plot 1: Fitness vs feature count with local optima highlighted ───────────

def plot_fitness_vs_features(df: pd.DataFrame, lo_mask: np.ndarray,
                             ax: plt.Axes):
    normal = df[~lo_mask]
    optima = df[lo_mask]

    ax.scatter(normal["num_features"], normal["mean_accuracy"],
               s=3, alpha=0.15, color="steelblue", label="regular", rasterized=True)
    ax.scatter(optima["num_features"], optima["mean_accuracy"],
               s=18, alpha=0.8, color="red", zorder=5,
               label=f"local optima ({len(optima)})", edgecolors="darkred",
               linewidths=0.3)

    grouped = df.groupby("num_features")["mean_accuracy"].mean()
    ax.plot(grouped.index, grouped.values, "k-", linewidth=1.5, label="mean")

    ax.set_xlabel("Number of active features")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Fitness vs Feature Count")
    ax.legend(fontsize="x-small", markerscale=1.5)


# ── Plot 2: Fitness distribution with local-optima overlay ───────────────────

def plot_fitness_distribution(df: pd.DataFrame, lo_mask: np.ndarray,
                              ax: plt.Axes):
    ax.hist(df["mean_accuracy"], bins=80, color="steelblue",
            edgecolor="white", linewidth=0.4, alpha=0.6, label="all")
    ax.hist(df.loc[lo_mask, "mean_accuracy"], bins=80, color="red",
            edgecolor="white", linewidth=0.4, alpha=0.7, label="local optima")
    ax.axvline(df["mean_accuracy"].max(), color="green", ls="--",
               label=f'global max = {df["mean_accuracy"].max():.4f}')
    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Fitness Distribution")
    ax.legend(fontsize="x-small")


# ── Plot 3: Fitness-Distance Correlation (FDC) ──────────────────────────────

def plot_fdc(df: pd.DataFrame, lo_mask: np.ndarray,
             n_features: int, ax: plt.Axes):
    global_best_idx = df.loc[df["mean_accuracy"].idxmax(), "index"]
    distances = df["index"].apply(lambda x: hamming_distance(x, global_best_idx))

    normal = df[~lo_mask]
    optima = df[lo_mask]
    d_normal = distances[~lo_mask]
    d_optima = distances[lo_mask]

    ax.scatter(d_normal, normal["mean_accuracy"],
               s=3, alpha=0.15, color="steelblue", rasterized=True, label="regular")
    ax.scatter(d_optima, optima["mean_accuracy"],
               s=18, alpha=0.8, color="red", zorder=5, edgecolors="darkred",
               linewidths=0.3, label="local optima")

    corr = np.corrcoef(distances, df["mean_accuracy"])[0, 1]
    ax.set_xlabel(f"Hamming distance to global optimum")
    ax.set_ylabel("Mean accuracy")
    ax.set_title(f"Fitness-Distance Correlation  (r = {corr:.3f})")
    ax.legend(fontsize="x-small", markerscale=1.5)

    ax.set_xticks(range(0, n_features + 1))


# ── Plot 4: Local-optima per feature count + summary stats ───────────────────

def plot_optima_summary(df: pd.DataFrame, lo_mask: np.ndarray,
                        n_features: int, ax: plt.Axes):
    total_by_k = df.groupby("num_features").size()
    optima_by_k = df[lo_mask].groupby("num_features").size()
    ks = range(1, n_features + 1)

    total_vals = [total_by_k.get(k, 0) for k in ks]
    optima_vals = [optima_by_k.get(k, 0) for k in ks]
    pct = [100.0 * o / t if t > 0 else 0 for o, t in zip(optima_vals, total_vals)]

    color_all = "steelblue"
    color_lo = "red"

    bars_all = ax.bar(list(ks), total_vals, color=color_all, alpha=0.4,
                      label="total combinations")
    bars_lo = ax.bar(list(ks), optima_vals, color=color_lo, alpha=0.7,
                     label="local optima")

    ax2 = ax.twinx()
    ax2.plot(list(ks), pct, "ko-", markersize=4, linewidth=1.2, label="% local optima")
    ax2.set_ylabel("% local optima", fontsize=12)
    ax2.set_ylim(0, max(pct) * 1.3 if max(pct) > 0 else 10)

    ax.set_xlabel("Number of active features")
    ax.set_ylabel("Count")
    ax.set_title("Local Optima Breakdown by Feature Count")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2,
              fontsize="x-small", loc="upper left")


# ── Plot 5: 3D t-SNE landscape surface ────────────────────────────────────────

def _index_to_bitvectors(indices: np.ndarray, n_features: int) -> np.ndarray:
    """Convert an array of integer indices to an (n, n_features) binary matrix."""
    bits = np.zeros((len(indices), n_features), dtype=np.float32)
    for b in range(n_features):
        bits[:, b] = (indices >> b) & 1
    return bits


def plot_3d_tsne(df: pd.DataFrame, lo_mask: np.ndarray,
                 n_features: int, stem: str, n_lo: int,
                 save_dir: str | None, max_points: int = 15000):
    """Produce a standalone 3D figure: t-SNE x/y, fitness z.

    For large landscapes the surface triangulation can exhaust memory, so we
    subsample down to *max_points* while always keeping every local optimum.
    """
    if len(df) > max_points:
        lo_idx_set = set(np.where(lo_mask)[0])
        non_lo = np.array([i for i in range(len(df)) if i not in lo_idx_set])
        rng = np.random.default_rng(42)
        keep_non_lo = rng.choice(
            non_lo, size=max_points - len(lo_idx_set), replace=False)
        keep = np.sort(np.concatenate([np.array(list(lo_idx_set)), keep_non_lo]))
        df_sub = df.iloc[keep].reset_index(drop=True)
        lo_mask_sub = np.isin(keep, list(lo_idx_set))
        print(f"    Subsampled {len(df)} -> {len(df_sub)} points for 3D plot "
              f"(all {len(lo_idx_set)} local optima retained)")
    else:
        df_sub = df
        lo_mask_sub = lo_mask

    print("    Running t-SNE (metric=hamming) ...")
    bit_matrix = _index_to_bitvectors(df_sub["index"].values, n_features)
    perplexity = min(30, len(df_sub) - 1)
    embedding = TSNE(
        n_components=2,
        metric="hamming",
        perplexity=perplexity,
        random_state=42,
        init="random",
    ).fit_transform(bit_matrix)

    x = embedding[:, 0]
    y = embedding[:, 1]
    z = df_sub["mean_accuracy"].values

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_trisurf(
        x, y, z,
        cmap="viridis", alpha=0.6, edgecolor="none", linewidth=0,
    )

    lo_idx = np.where(lo_mask_sub)[0]
    if len(lo_idx) > 0:
        ax.scatter(
            x[lo_idx], y[lo_idx], z[lo_idx],
            s=40, color="red", edgecolors="darkred", linewidths=0.5,
            zorder=10, label=f"local optima ({len(lo_idx)})",
            depthshade=False,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("Mean accuracy")
    ax.set_title(
        f"{stem}  —  3D t-SNE Landscape\n"
        f"(N={n_features},  {n_lo} local optima)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize="small", loc="upper left")

    fig.colorbar(surf, ax=ax, shrink=0.5, label="Mean accuracy")
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{stem}_3d.png")
        fig.savefig(out, dpi=150)
        print(f"    Saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Per-file driver ──────────────────────────────────────────────────────────

def visualise(csv_path: str, save_dir: str | None):
    df = pd.read_csv(csv_path)
    for col in ("index", "num_features", "mean_accuracy", "mean_time"):
        if col not in df.columns:
            sys.exit(f"Missing expected column '{col}' in {csv_path}")

    n_features = int(np.log2(df["index"].max() + 1))
    print(f"  {os.path.basename(csv_path)}: {len(df)} combinations, "
          f"N={n_features}")

    lo_mask = find_local_optima(df, n_features, include_plateaus=False)
    n_lo = lo_mask.sum()
    print(f"    Local optima (strict peaks): {n_lo} "
          f"({100*n_lo/len(df):.2f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    stem = os.path.basename(csv_path).replace(".csv", "")
    fig.suptitle(f"{stem}   (N={n_features},  {n_lo} local optima)",
                 fontsize=16, fontweight="bold")

    plot_fitness_vs_features(df, lo_mask, axes[0, 0])
    plot_fitness_distribution(df, lo_mask, axes[0, 1])
    plot_fdc(df, lo_mask, n_features, axes[1, 0])
    plot_optima_summary(df, lo_mask, n_features, axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{stem}.png")
        fig.savefig(out, dpi=150)
        print(f"    Saved to {out}")
        plt.close(fig)
    else:
        plt.show()

    plot_3d_tsne(df, lo_mask, n_features, stem, n_lo, save_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise fitness landscapes with local optima highlighted")
    parser.add_argument("csv", nargs="+",
                        help="One or more CSVs exported by the C++ program")
    parser.add_argument("--save", default=None,
                        help="Directory to save PNGs instead of showing")
    args = parser.parse_args()

    for csv_path in args.csv:
        print(f"Processing {csv_path} ...")
        visualise(csv_path, args.save)


if __name__ == "__main__":
    main()
