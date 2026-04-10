#!/usr/bin/env python3
"""
Animate NSGA-II population evolution on a 2D hinged-bitstring heatmap.

Two panels:
  Left  - 2D heatmap of the fitness landscape with current population dots.
  Right - Objective-space view with persistent per-generation trail.

Usage:
    python animate_nsga2.py --snapshots output/breast-w_snapshots.csv \
                            --landscape output/breast-w.csv --save figures
    python animate_nsga2.py --snapshots output/triangle_snapshots.csv \
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
    """Build a 2D fitness grid by splitting the bitstring index in two halves."""
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

def build_animation(snapshots: pd.DataFrame, landscape: pd.DataFrame,
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

    # Objective-space axis limits
    obj_acc_lo = snapshots["accuracy"].min()
    obj_acc_hi = snapshots["accuracy"].max()
    obj_time_lo = snapshots["time"].min()
    obj_time_hi = snapshots["time"].max()
    obj_acc_pad = (obj_acc_hi - obj_acc_lo) * 0.08 or 0.01
    obj_time_pad = (obj_time_hi - obj_time_lo) * 0.08 or 0.01

    grouped = {g: df for g, df in snapshots.groupby("generation")}

    # -- Figure layout ---------------------------------------------------------
    fig, (ax_map, ax_obj) = plt.subplots(
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
    ax_map.set_title("Population on Landscape", fontsize=12)

    scat_dom = ax_map.scatter([], [], s=18, alpha=0.45, zorder=4,
                               edgecolors="none", color="dodgerblue")
    scat_par = ax_map.scatter([], [], s=90, zorder=6,
                               facecolors="red", edgecolors="white",
                               linewidths=1.8)

    # -- Right panel: objective space ------------------------------------------
    ax_obj.set_xlabel("Objective 2  (time / complexity)")
    ax_obj.set_ylabel("Accuracy  (objective 1)")
    ax_obj.set_title("Objective Space", fontsize=12)
    ax_obj.set_xlim(obj_time_lo - obj_time_pad, obj_time_hi + obj_time_pad)
    ax_obj.set_ylim(obj_acc_lo - obj_acc_pad, obj_acc_hi + obj_acc_pad)
    ax_obj.grid(True, alpha=0.3)
    ax_obj.set_facecolor("#f7f7f7")

    scat_obj_dom = ax_obj.scatter([], [], s=18, alpha=0.4, zorder=3,
                                  edgecolors="none", color="dodgerblue")
    scat_obj_par = ax_obj.scatter([], [], s=70, zorder=6,
                                  facecolors="red", edgecolors="white",
                                  linewidths=1.4, marker="D")
    line_pareto, = ax_obj.plot([], [], "r-", lw=1.8, alpha=0.6, zorder=5)

    # Persistent generation trail on objective space
    max_gen = generations[-1]
    trail_cmap = plt.cm.plasma
    trail_norm = plt.Normalize(vmin=0, vmax=max_gen)

    scat_trail_obj = ax_obj.scatter([], [], s=20, alpha=0.5, zorder=2,
                                     edgecolors="black", linewidths=0.2)

    cbar_trail = fig.colorbar(
        cm.ScalarMappable(norm=trail_norm, cmap="plasma"),
        ax=ax_obj, shrink=0.82, pad=0.02)
    cbar_trail.set_label("Generation", fontsize=9)

    trail_time_list: list[np.ndarray] = []
    trail_acc_list: list[np.ndarray] = []
    trail_gen_list: list[np.ndarray] = []

    def update(frame_idx):
        gen = frame_gens[frame_idx]
        gdf = grouped[gen]

        ranks = gdf["rank"].values
        pareto_mask = ranks == 0

        pareto = gdf[pareto_mask]
        dominated = gdf[~pareto_mask]

        # Accumulate trail
        trail_time_list.append(gdf["time"].values)
        trail_acc_list.append(gdf["accuracy"].values)
        trail_gen_list.append(np.full(len(gdf), gen))

        all_trail_time = np.concatenate(trail_time_list)
        all_trail_acc = np.concatenate(trail_acc_list)
        all_trail_gen = np.concatenate(trail_gen_list)
        trail_colors = trail_cmap(trail_norm(all_trail_gen))

        scat_trail_obj.set_offsets(
            np.column_stack([all_trail_time, all_trail_acc]))
        scat_trail_obj.set_facecolors(trail_colors)

        # -- Heatmap panel: current generation only ----------------------------
        if len(dominated) > 0:
            scat_dom.set_offsets(
                np.column_stack([dominated["gx"].values,
                                 dominated["gy"].values]))
        else:
            scat_dom.set_offsets(np.empty((0, 2)))

        if len(pareto) > 0:
            scat_par.set_offsets(
                np.column_stack([pareto["gx"].values,
                                 pareto["gy"].values]))
        else:
            scat_par.set_offsets(np.empty((0, 2)))

        # -- Objective-space panel ---------------------------------------------
        if len(dominated) > 0:
            scat_obj_dom.set_offsets(
                np.column_stack([dominated["time"].values,
                                 dominated["accuracy"].values]))
        else:
            scat_obj_dom.set_offsets(np.empty((0, 2)))

        if len(pareto) > 0:
            scat_obj_par.set_offsets(
                np.column_stack([pareto["time"].values,
                                 pareto["accuracy"].values]))
            psorted = pareto.sort_values("time")
            line_pareto.set_data(psorted["time"].values,
                                 psorted["accuracy"].values)
        else:
            scat_obj_par.set_offsets(np.empty((0, 2)))
            line_pareto.set_data([], [])

        pareto_size = int(pareto_mask.sum())
        title.set_text(
            f"{stem}  |  Generation {gen}/{generations[-1]}"
            f"  |  Pareto front: {pareto_size}"
            f"  |  Pop: {len(gdf)}")

        return (scat_dom, scat_par, scat_obj_dom, scat_obj_par,
                line_pareto, scat_trail_obj)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_gens),
        interval=1000 // fps, blit=False, repeat_delay=2000)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{stem}_evolution.gif")
        print(f"  Rendering {len(frame_gens)} frames to {out_path} ...")
        anim.save(out_path, writer="pillow", fps=fps, dpi=110)
        print(f"    Saved to {out_path}")
        plt.close(fig)
    else:
        plt.show()


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate NSGA-II evolution on the fitness landscape")
    parser.add_argument("--snapshots", required=True,
                        help="Path to *_snapshots.csv from the NSGA-II run")
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

    print(f"Loading snapshots from {args.snapshots} ...")
    snapshots = pd.read_csv(args.snapshots)
    n_gens = snapshots["generation"].nunique()
    pop_size = len(snapshots[snapshots["generation"] == 0])
    print(f"  {n_gens} generations, pop_size={pop_size}, "
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
    build_animation(snapshots, landscape, stem, args.save, args.fps)


if __name__ == "__main__":
    main()
