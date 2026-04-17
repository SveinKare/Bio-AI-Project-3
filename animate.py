import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import glob
import re

experiments = [
    {"name": "triangle",  "landscape": "data/triangle.csv",         "bits": 16, "fit_col": 1},
    {"name": "breast",    "landscape": "data/landscape_breast.csv", "bits": 9,  "fit_col": 3},
    {"name": "credit",    "landscape": "data/landscape_credit.csv", "bits": 15, "fit_col": 3},
    {"name": "letter",    "landscape": "data/landscape_letter.csv", "bits": 16, "fit_col": 3},
]

def animate_experiment(exp):
    bits = exp["bits"]
    upper = bits // 2
    lower = bits - upper
    grid_h = 1 << upper
    grid_w = 1 << lower

    # Build landscape grid
    df = pd.read_csv(exp["landscape"])
    fitness = np.zeros(1 << bits)
    for _, row in df.iterrows():
        idx = int(row.iloc[0])
        fitness[idx] = row.iloc[exp["fit_col"]]

    grid = np.zeros((grid_h, grid_w))
    for i in range(len(fitness)):
        grid[i >> lower, i & ((1 << lower) - 1)] = fitness[i]

    # Find and sort generation files
    files = glob.glob(f"anim/{exp['name']}_gen_*.csv")
    def gen_num(f):
        return int(re.search(r'gen_(\d+)', f).group(1))
    files.sort(key=gen_num)

    fig, ax = plt.subplots(figsize=(8, 8))
    landscape_img = ax.imshow(grid, cmap="viridis", origin="lower", aspect="equal")
    overlay_img = ax.imshow(np.zeros((grid_h, grid_w, 4)), origin="lower", aspect="equal")
    title = ax.set_title("")
    ax.set_xlabel(f"Lower {lower} bits")
    ax.set_ylabel(f"Upper {upper} bits")

    def update(frame):
        pop = pd.read_csv(files[frame])
        pop_grid = np.zeros((grid_h, grid_w))
        for _, row in pop.iterrows():
            idx = int(row["id"])
            pop_grid[idx >> lower, idx & ((1 << lower) - 1)] += 1
        pop_smooth = gaussian_filter(pop_grid, sigma=2)

        pop_alpha = np.where(pop_smooth > 0.01, np.clip(pop_smooth / pop_smooth.max(), 0.15, 0.7), 0)
        red_overlay = np.zeros((grid_h, grid_w, 4))
        red_overlay[..., 0] = 1.0
        red_overlay[..., 3] = pop_alpha
        overlay_img.set_data(red_overlay)

        gen = gen_num(files[frame])
        title.set_text(f"{exp['name']} — Generation {gen}")
        return [overlay_img, title]

    anim = FuncAnimation(fig, update, frames=len(files), interval=500, blit=True)
    anim.save(f"anim_{exp['name']}.gif", writer="pillow", fps=2)
    plt.close()
    print(f"Saved anim_{exp['name']}.gif")

for exp in experiments:
    animate_experiment(exp)
