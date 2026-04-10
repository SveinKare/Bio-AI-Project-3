#!/usr/bin/env python3
"""
Animate Binary PSO swarm evolution on a 2D hinged-bitstring heatmap.

Two panels:
  Left  - 2D heatmap of the fitness landscape with current swarm particles.
  Right - Live fitness convergence curve (global best + mean).

Usage:
    python animate_pso.py --snapshots output/pso-breast-w_snapshots.csv \
                          --gens output/pso-breast-w_gens.csv \
                          --landscape output/breast-w.csv --save figures
    python animate_pso.py --snapshots output/pso-triangle_snapshots.csv \
                          --gens output/pso-triangle_gens.csv \
                          --triangle --tri-n 16 --tri-m 1 --tri-s 4 \
                          --save figures
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "white",
    "font.size": 11,
})


# -- Triangle landscape generator ---------------------------------------------

def triangle_fitness(b: int, m: int, s: int) -> float:
    if b == 0:
        return 0.0
    ceil_bs = (b + s - 1) // s
    if ceil_bs % 2 == 1:
        return float(m * s) if b % s == 0 else float(m * (b % s))
    return float(m * (ceil_bs * s - b))


def generate_triangle_df(n: int, m: int, s: int) -> pd.DataFrame:
    rows = []
    for idx in range(1 << n):
        b = bin(idx).count("1")
        rows.append((idx, format(idx, f"0{n}b"), b,
                      triangle_fitness(b, m, s), 0.0))
    return pd.DataFrame(rows, columns=[
        "index", "bitmask", "num_features", "mean_accuracy", "mean_time"])


# -- Bitstring grid helpers ----------------------------------------------------

def _build_heatmap(landscape: pd.DataFrame, n_features: int):
    n_x = n_features // 2
    n_y = n_features - n_x
    width = 1 << n_x
    height = 1 << n_y
    x_mask = width - 1

    Z = np.full((height, width), np.nan)
    for idx, acc in zip(landscape["index"].values,
                        landscape["mean_accuracy"].values):
        x = int(idx) & x_mask
        y = int(idx) >> n_x
        Z[y, x] = acc

    return Z, n_x, n_y, width, height, x_mask


def _index_to_grid(idx: int, n_x: int, x_mask: int):
    return idx & x_mask, idx >> n_x


# -- Build the animation ------------------------------------------------------

def build_animation(snapshots: pd.DataFrame, gens: pd.DataFrame,
                    landscape: pd.DataFrame,
                    stem: str, save_dir: str | None, fps: int = 10):

    generations = sorted(snapshots["generation"].unique())
    n_gens = len(generations)
    n_features = int(np.log2(landscape["index"].max() + 1))

    Z, n_x, n_y, width, height, x_mask = _build_heatmap(landscape, n_features)

    # Precompute grid coordinates
    all_gx = np.empty(len(snapshots), dtype=float)
    all_gy = np.empty(len(snapshots), dtype=float)
    for i, idx in enumerate(snapshots["index"].values):
        all_gx[i], all_gy[i] = _index_to_grid(int(idx), n_x, x_mask)
    snapshots = snapshots.copy()
    snapshots["gx"] = all_gx
    snapshots["gy"] = all_gy

    # Subsample frames for reasonable GIF size
    frame_step = max(1, n_gens // 200)
    frame_gens = list(generations[::frame_step])
    if frame_gens[-1] != generations[-1]:
        frame_gens.append(generations[-1])

    grouped = {g: df for g, df in snapshots.groupby("generation")}

    # Gens data indexed by generation
    gens_indexed = gens.set_index("generation")

    # -- Figure layout ---------------------------------------------------------
    fig, (ax_map, ax_conv) = plt.subplots(
        1, 2, figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(top=0.90, wspace=0.28)
    title = fig.suptitle("", fontsize=14, fontweight="bold")

    # -- Left panel: static heatmap --------------------------------------------
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    Z_plot = np.where(np.isnan(Z), z_min, Z)

    ax_map.imshow(
        Z_plot, origin="lower", cmap="viridis",
        vmin=z_min, vmax=z_max, aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(z_min, z_max), cmap="viridis"),
        ax=ax_map, shrink=0.82, pad=0.02)
    cbar.set_label("Fitness (accuracy)", fontsize=10)
    ax_map.set_xlabel(f"Lower bits  (bits 0..{n_x-1})", fontsize=10)
    ax_map.set_ylabel(f"Upper bits  (bits {n_x}..{n_features-1})", fontsize=10)
    ax_map.set_title("Swarm on Landscape", fontsize=12)

    scat_swarm = ax_map.scatter([], [], s=18, alpha=0.45, zorder=4,
                                 edgecolors="none", color="dodgerblue")
    scat_best = ax_map.scatter([], [], s=150, zorder=6,
                                facecolors="gold", edgecolors="black",
                                linewidths=1.5, marker="*")

    # -- Right panel: convergence curves ---------------------------------------
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Value")
    ax_conv.set_title("Fitness Convergence", fontsize=12)
    ax_conv.set_xlim(0, generations[-1])

    gbest_vals = gens["hypervolume"].values
    mean_acc_vals = gens["mean_accuracy"].values
    best_acc_vals = gens["best_accuracy"].values

    y_lo = min(mean_acc_vals.min(), gbest_vals.min())
    y_hi = max(best_acc_vals.max(), gbest_vals.max())
    y_pad = (y_hi - y_lo) * 0.08 or 0.01
    ax_conv.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.set_facecolor("#f7f7f7")

    line_gbest, = ax_conv.plot([], [], "r-", lw=2.5, label="global best fitness")
    line_best_acc, = ax_conv.plot([], [], "g--", lw=1.5, alpha=0.8,
                                   label="best accuracy (raw)")
    line_mean_acc, = ax_conv.plot([], [], "b:", lw=1.5, alpha=0.6,
                                   label="mean accuracy")
    ax_conv.legend(fontsize="x-small", loc="lower right")

    # Track convergence history for the line plots
    conv_iters: list[int] = []
    conv_gbest: list[float] = []
    conv_best_acc: list[float] = []
    conv_mean_acc: list[float] = []

    def update(frame_idx):
        gen = frame_gens[frame_idx]
        gdf = grouped[gen]

        ranks = gdf["rank"].values
        best_mask = ranks == 0
        best_particles = gdf[best_mask]
        rest = gdf[~best_mask]

        # -- Heatmap panel: current swarm --------------------------------------
        if len(rest) > 0:
            scat_swarm.set_offsets(
                np.column_stack([rest["gx"].values, rest["gy"].values]))
        else:
            scat_swarm.set_offsets(np.empty((0, 2)))

        if len(best_particles) > 0:
            scat_best.set_offsets(
                np.column_stack([best_particles["gx"].values,
                                 best_particles["gy"].values]))
        else:
            scat_best.set_offsets(np.empty((0, 2)))

        # -- Convergence panel: update curves ----------------------------------
        if gen in gens_indexed.index:
            row = gens_indexed.loc[gen]
            conv_iters.append(gen)
            conv_gbest.append(row["hypervolume"])
            conv_best_acc.append(row["best_accuracy"])
            conv_mean_acc.append(row["mean_accuracy"])

            line_gbest.set_data(conv_iters, conv_gbest)
            line_best_acc.set_data(conv_iters, conv_best_acc)
            line_mean_acc.set_data(conv_iters, conv_mean_acc)

        best_fit = conv_gbest[-1] if conv_gbest else 0
        title.set_text(
            f"{stem}  |  Iteration {gen}/{generations[-1]}"
            f"  |  Best fitness: {best_fit:.4f}"
            f"  |  Swarm: {len(gdf)}")

        return (scat_swarm, scat_best,
                line_gbest, line_best_acc, line_mean_acc)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_gens),
        interval=1000 // fps, blit=False, repeat_delay=2000)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"pso-{stem}_evolution.gif")
        print(f"  Rendering {len(frame_gens)} frames to {out_path} ...")
        anim.save(out_path, writer="pillow", fps=fps, dpi=110)
        print(f"    Saved to {out_path}")
        plt.close(fig)
    else:
        plt.show()


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate Binary PSO swarm evolution on the fitness landscape")
    parser.add_argument("--snapshots", required=True,
                        help="Path to *_snapshots.csv from the PSO run")
    parser.add_argument("--gens", required=True,
                        help="Path to *_gens.csv from the PSO run")
    parser.add_argument("--landscape", default=None,
                        help="Path to landscape CSV (exported by C++ program)")
    parser.add_argument("--triangle", action="store_true",
                        help="Use synthetic triangle landscape")
    parser.add_argument("--tri-n", type=int, default=16)
    parser.add_argument("--tri-m", type=int, default=1)
    parser.add_argument("--tri-s", type=int, default=4)
    parser.add_argument("--save", default=None,
                        help="Directory to save GIF")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second (default 10)")
    args = parser.parse_args()

    if not os.path.exists(args.snapshots):
        sys.exit(f"Snapshots file not found: {args.snapshots}")
    if not os.path.exists(args.gens):
        sys.exit(f"Gens file not found: {args.gens}")

    print(f"Loading snapshots from {args.snapshots} ...")
    snapshots = pd.read_csv(args.snapshots)
    gens = pd.read_csv(args.gens)
    n_gens = snapshots["generation"].nunique()
    swarm_size = len(snapshots[snapshots["generation"] == 0])
    print(f"  {n_gens} iterations, swarm_size={swarm_size}, "
          f"total rows={len(snapshots)}")

    if args.triangle:
        landscape = generate_triangle_df(args.tri_n, args.tri_m, args.tri_s)
        stem = f"triangle_n{args.tri_n}_m{args.tri_m}_s{args.tri_s}"
    elif args.landscape:
        if not os.path.exists(args.landscape):
            sys.exit(f"Landscape file not found: {args.landscape}")
        landscape = pd.read_csv(args.landscape)
        stem = os.path.basename(args.landscape).replace(".csv", "")
    else:
        parser.error("Provide --landscape <csv> or --triangle")

    print(f"Landscape: {stem}, {len(landscape)} entries")
    build_animation(snapshots, gens, landscape, stem, args.save, args.fps)


if __name__ == "__main__":
    main()
