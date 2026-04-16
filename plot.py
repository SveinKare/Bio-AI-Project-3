import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

experiments = [
    {"name": "triangle",      "display": "Triangle",       "landscape": "triangle.csv",         "pop": "pop_triangle.csv", "bits": 16, "fit_col": 1},
    {"name": "breast",        "display": "Breast Cancer",  "landscape": "landscape_breast.csv", "pop": "pop_breast.csv",   "bits": 9,  "fit_col": 3},
    {"name": "credit",        "display": "Credit",         "landscape": "landscape_credit.csv", "pop": "pop_credit.csv",   "bits": 15, "fit_col": 3},
    {"name": "letter",        "display": "Letter",         "landscape": "landscape_letter.csv", "pop": "pop_letter.csv",   "bits": 16, "fit_col": 3},
]

def plot_heatmap(exp):
    bits = exp["bits"]
    upper = bits // 2
    lower = bits - upper
    grid_h = 1 << upper
    grid_w = 1 << lower

    df = pd.read_csv(exp["landscape"])
    fitness = np.zeros(1 << bits)
    for _, row in df.iterrows():
        idx = int(row.iloc[0])
        fitness[idx] = row.iloc[exp["fit_col"]]

    grid = np.zeros((grid_h, grid_w))
    for i in range(len(fitness)):
        grid[i >> lower, i & ((1 << lower) - 1)] = fitness[i]

    pop = pd.read_csv(exp["pop"])
    pop_grid = np.zeros((grid_h, grid_w))
    for _, row in pop.iterrows():
        idx = int(row["id"])
        pop_grid[idx >> lower, idx & ((1 << lower) - 1)] += 1
    pop_smooth = gaussian_filter(pop_grid, sigma=2)

    # Create circular mask for each population cluster
    from scipy.ndimage import label, center_of_mass, binary_dilation
    binary = pop_smooth > 0.01 * pop_smooth.max()
    labeled, n_clusters = label(binary)
    centers = center_of_mass(pop_smooth, labeled, range(1, n_clusters + 1))

    circular_mask = np.zeros_like(pop_smooth)
    Y, X = np.ogrid[:grid_h, :grid_w]
    for i, (cy, cx) in enumerate(centers):
        cluster_mask = labeled == (i + 1)
        # Radius based on cluster size
        area = cluster_mask.sum()
        radius = max(np.sqrt(area / np.pi), 2)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        circle = dist <= radius
        circular_mask = np.maximum(circular_mask, pop_smooth * circle)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap="viridis", origin="lower", aspect="equal")

    pop_alpha = np.where(circular_mask > 0.01, np.clip(circular_mask / circular_mask.max(), 0.15, 0.7), 0)
    red_overlay = np.zeros((*circular_mask.shape, 4))
    red_overlay[..., 0] = 1.0
    red_overlay[..., 3] = pop_alpha
    ax.imshow(red_overlay, origin="lower", aspect="equal")

    ax.set_xlabel(f"Lower {lower} bits")
    ax.set_ylabel(f"Upper {upper} bits")
    ax.set_title(f"{exp['name']} (N={bits})")
    plt.tight_layout()
    plt.savefig(f"heatmap_{exp['name'].lower().replace(' ', '_')}.png", dpi=200)
    plt.show()

def plot_histograms(exp):
    df = pd.read_csv(exp["landscape"])
    landscape_fitness = df.iloc[:, exp["fit_col"]].values

    pop = pd.read_csv(exp["pop"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.hist(landscape_fitness, bins=100, color="steelblue", alpha=0.7)
    ax1.set_ylabel("Landscape count")
    ax1.set_title(f"{exp['name']} — Fitness Distribution")

    if "niche" in pop.columns:
        niche_ids = sorted(pop["niche"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(niche_ids), 10)))
        for niche_id, color in zip(niche_ids, colors):
            niche_fit = pop[pop["niche"] == niche_id]["fitness"].values
            ax2.hist(niche_fit, bins=100, color=color, alpha=0.7, label=f"Niche {niche_id}")
        ax2.legend()
    else:
        ax2.hist(pop["fitness"].values, bins=50, color="red", alpha=0.7)

    ax2.set_xlabel("Fitness")
    ax2.set_ylabel("Population count")

    plt.tight_layout()
    plt.savefig(f"hist_{exp['name'].lower().replace(' ', '_')}.png", dpi=200)
    plt.show()

def plot_diversity(exp):
    df = pd.read_csv(f"stats_{exp['name']}.csv")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["generation"], df["avg_hamming"], color="steelblue", label="Avg hamming distance")
    ax1.plot(df["generation"], df["entropy"], color="purple", linestyle="-.", label="Entropy")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Diversity")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["generation"], df["max_fitness"], color="red", label="Max fitness")
    ax2.plot(df["generation"], df["avg_fitness"], color="orange", linestyle="--", label="Avg fitness")
    ax2.set_ylabel("Fitness")
    ax2.legend(loc="upper right")

    plt.title(f"{exp['display']} — Diversity & Entropy vs Fitness")
    plt.tight_layout()
    plt.savefig(f"diversity_{exp['name']}.png", dpi=200)
    plt.show()

for exp in experiments:
    #plot_heatmap(exp)
    #plot_histograms(exp)
    plot_diversity(exp)
