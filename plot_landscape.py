#!/usr/bin/env python3
"""
Visualise a feature-selection fitness landscape exported as CSV,
with every local optimum explicitly highlighted.

Usage:
    python plot_landscape.py <landscape.csv> [--save <output_dir>]
    python plot_landscape.py output/*.csv --save figures
    python plot_landscape.py --triangle --save figures

The CSV is produced by:  ./build/main <file.h5> --export landscape.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for projection='3d'
import numpy as np
import pandas as pd
import seaborn as sns

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
    min_index = int(df["index"].min())
    max_index = (1 << n_features) - 1
    is_optimum = np.zeros(len(df), dtype=bool)

    for row_pos, (idx, fit) in enumerate(
            zip(df["index"].values, df["mean_accuracy"].values)):
        dominated = False
        strictly_better = False
        for bit in range(n_features):
            nb = idx ^ (1 << bit)
            if nb < min_index or nb > max_index:
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
    ax.set_xlabel("Hamming distance to global optimum")
    ax.set_ylabel("Mean accuracy")
    ax.set_title(f"Fitness-Distance Correlation  (r = {corr:.3f})")
    ax.legend(fontsize="x-small", markerscale=1.5)

    ax.set_xticks(range(0, n_features + 1))


# ── Plot 4: Local-optima per feature count + summary stats ───────────────────

def plot_optima_summary(df: pd.DataFrame, lo_mask: np.ndarray,
                        n_features: int, ax: plt.Axes):
    min_k = int(df["num_features"].min())
    total_by_k = df.groupby("num_features").size()
    optima_by_k = df[lo_mask].groupby("num_features").size()
    ks = range(min_k, n_features + 1)

    total_vals = [total_by_k.get(k, 0) for k in ks]
    optima_vals = [optima_by_k.get(k, 0) for k in ks]
    pct = [100.0 * o / t if t > 0 else 0 for o, t in zip(optima_vals, total_vals)]

    ax.bar(list(ks), total_vals, color="steelblue", alpha=0.4,
           label="total combinations")
    ax.bar(list(ks), optima_vals, color="red", alpha=0.7,
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


# ── Plot 5: 3D Hinged Bitstring Map ──────────────────────────────────────────

def _gray_decode(g: int) -> int:
    """Convert a Gray-code value back to the binary integer it encodes."""
    n = g
    mask = n >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n


def _build_bitstring_grid(df: pd.DataFrame, n_features: int):
    """Map every bitstring index onto a 2D grid via hinged Gray-code ordering."""
    n_x = n_features // 2
    n_y = n_features - n_x
    width = 1 << n_x
    height = 1 << n_y
    x_mask = width - 1

    fitness_lookup = dict(zip(df["index"].values, df["mean_accuracy"].values))
    lo_set = set(df.loc[df["_is_lo"], "index"].values) if "_is_lo" in df.columns else set()

    Z = np.full((height, width), np.nan)
    lo_grid = np.zeros((height, width), dtype=bool)

    for idx, acc in fitness_lookup.items():
        x_bits = idx & x_mask
        y_bits = idx >> n_x
        gx = _gray_decode(x_bits)
        gy = _gray_decode(y_bits)
        Z[gy, gx] = acc
        if idx in lo_set:
            lo_grid[gy, gx] = True

    gx_coords = np.arange(width)
    gy_coords = np.arange(height)

    return Z, lo_grid, gx_coords, gy_coords, n_x, n_y


def plot_3d_bitstring_map(df: pd.DataFrame, lo_mask: np.ndarray,
                          n_features: int, stem: str, n_lo: int,
                          save_dir: str | None):
    """3D surface on a hinged bitstring map grid, with local optima in red."""
    df = df.copy()
    df["_is_lo"] = lo_mask

    Z, lo_grid, gx, gy, n_x, n_y = _build_bitstring_grid(df, n_features)

    GX, GY = np.meshgrid(gx, gy)
    Z_plot = np.where(np.isnan(Z), 0.0, Z)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    colors = cm.viridis(norm(Z_plot))
    colors[..., 3] = 0.8

    ax.plot_surface(
        GX, GY, Z_plot,
        facecolors=colors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
        shade=True,
    )

    lo_gy, lo_gx = np.where(lo_grid)
    if len(lo_gx) > 0:
        lo_z = Z[lo_gy, lo_gx]
        ax.scatter(
            lo_gx, lo_gy, lo_z + (np.nanmax(Z) - np.nanmin(Z)) * 0.01,
            s=50, color="red", edgecolors="darkred", linewidths=0.6,
            zorder=10, label=f"local optima ({n_lo})",
            depthshade=False,
        )

    ax.set_xlabel(f"Gray-code X  (bits 0..{n_x-1})")
    ax.set_ylabel(f"Gray-code Y  (bits {n_x}..{n_features-1})")
    ax.set_zlabel("Mean accuracy")
    ax.set_title(
        f"{stem}  --  3D Hinged Bitstring Map\n"
        f"(N={n_features},  {n_lo} local optima)",
        fontsize=14, fontweight="bold",
    )
    if n_lo > 0:
        ax.legend(fontsize="small", loc="upper left")

    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(mappable, ax=ax, shrink=0.5, label="Mean accuracy")
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{stem}_3d.png")
        fig.savefig(out, dpi=150)
        print(f"    Saved to {out}")
        plt.close(fig)
    else:
        plt.show()


# ── Triangle (synthetic) landscape generator ─────────────────────────────────

def triangle_fitness(b: int, m: int, s: int) -> float:
    """Compute the triangle function for popcount b, parameters m and s."""
    if b == 0:
        return 0.0
    ceil_bs = (b + s - 1) // s
    if ceil_bs % 2 == 1:
        if b % s == 0:
            return float(m * s)
        else:
            return float(m * (b % s))
    else:
        return float(m * (ceil_bs * s - b))


def generate_triangle_csv(n: int, m: int, s: int, csv_path: str) -> str:
    """Generate CSV for the full triangle landscape (2^n entries, index 0..2^n-1)."""
    rows = []
    for idx in range(1 << n):
        b = bin(idx).count("1")
        fit = triangle_fitness(b, m, s)
        bitmask = format(idx, f"0{n}b")
        rows.append((idx, bitmask, b, fit, 0.0))

    df = pd.DataFrame(rows, columns=["index", "bitmask", "num_features",
                                      "mean_accuracy", "mean_time"])
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Generated triangle landscape CSV: {csv_path}")
    print(f"    n={n}, m={m}, s={s}, entries={len(df)}")
    return csv_path


# ── Per-file driver ──────────────────────────────────────────────────────────

def visualise(csv_path: str, save_dir: str | None):
    df = pd.read_csv(csv_path)
    for col in ("index", "num_features", "mean_accuracy"):
        if col not in df.columns:
            sys.exit(f"Missing expected column '{col}' in {csv_path}")
    if "mean_time" not in df.columns:
        df["mean_time"] = 0.0

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

    plot_3d_bitstring_map(df, lo_mask, n_features, stem, n_lo, save_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise fitness landscapes with local optima highlighted")
    parser.add_argument("csv", nargs="*",
                        help="One or more CSVs exported by the C++ program")
    parser.add_argument("--save", default=None,
                        help="Directory to save PNGs instead of showing")
    parser.add_argument("--triangle", action="store_true",
                        help="Generate and visualise the synthetic triangle "
                             "landscape (n=16, m=1, s=4 by default)")
    parser.add_argument("--tri-n", type=int, default=16,
                        help="Triangle landscape: number of bits (default 16)")
    parser.add_argument("--tri-m", type=int, default=1,
                        help="Triangle landscape: m parameter (default 1)")
    parser.add_argument("--tri-s", type=int, default=4,
                        help="Triangle landscape: s parameter (default 4)")
    args = parser.parse_args()

    csv_files = list(args.csv) if args.csv else []

    if args.triangle:
        out_dir = args.save or "output"
        tri_csv = os.path.join(out_dir,
            f"triangle_n{args.tri_n}_m{args.tri_m}_s{args.tri_s}.csv")
        generate_triangle_csv(args.tri_n, args.tri_m, args.tri_s, tri_csv)
        csv_files.append(tri_csv)

    if not csv_files:
        parser.error("Provide at least one CSV or use --triangle")

    for csv_path in csv_files:
        print(f"Processing {csv_path} ...")
        visualise(csv_path, args.save)


if __name__ == "__main__":
    main()
