import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

experiments = [
    {"name": "triangle",      "display": "Triangle",       "landscape": "data/triangle.csv",         "pop": "pop_triangle.csv", "bits": 16, "fit_col": 1},
    {"name": "breast",        "display": "Breast Cancer",  "landscape": "data/landscape_breast.csv", "pop": "pop_breast.csv",   "bits": 9,  "fit_col": 3},
    {"name": "credit",        "display": "Credit",         "landscape": "data/landscape_credit.csv", "pop": "pop_credit.csv",   "bits": 15, "fit_col": 3},
    {"name": "letter",        "display": "Letter",         "landscape": "data/landscape_letter.csv", "pop": "pop_letter.csv",   "bits": 16, "fit_col": 3},
]

def plot_fitness_distribution(exp):
    """Overlay population fitness histogram on top of the landscape's fitness distribution."""
    df = pd.read_csv(exp["landscape"])
    landscape_fitness = df.iloc[:, exp["fit_col"]].values

    pop = pd.read_csv(exp["pop"])
    pop_fitness = pop["fitness"].values

    # Shared bin edges so the two histograms are directly comparable
    fmin = min(landscape_fitness.min(), pop_fitness.min())
    fmax = max(landscape_fitness.max(), pop_fitness.max())
    bins = np.linspace(fmin, fmax, 80)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Landscape distribution (how fitness is distributed across the whole search space)
    ax1.hist(landscape_fitness, bins=bins, color="steelblue", alpha=0.6,
             label=f"Landscape (N={len(landscape_fitness)})")
    ax1.set_xlabel("Fitness")
    ax1.set_ylabel("Landscape count", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Population distribution on a second y-axis so the small pop doesn't vanish
    ax2 = ax1.twinx()
    ax2.hist(pop_fitness, bins=bins, color="red", alpha=0.7,
             label=f"Population (N={len(pop_fitness)})")
    ax2.set_ylabel("Population count", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Mark where the population sits relative to the landscape's best
    ax1.axvline(landscape_fitness.max(), color="black", linestyle=":",
                linewidth=1, label="Landscape max")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"{exp['display']} — Population vs. Landscape Fitness Distribution")
    plt.tight_layout()
    plt.savefig(f"fitdist_{exp['name']}.png", dpi=200)
    plt.show()

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

def plot_grid_search():
    import pandas as pd
    import numpy as np

    landscapes = ["triangle", "breast", "credit", "letter"]
    bits = {"triangle": 16, "breast": 9, "credit": 15, "letter": 16}
    runs = pd.read_csv("experiment_runs.csv")

    rows = []
    for land in landscapes:
        opt = pd.read_csv(f"data/optima_{land}.csv")
        global_fitness = opt["fitness"].max()

        sub = runs[runs["landscape"] == land]

        # Best run for this landscape
        best_idx = sub["best_accuracy"].idxmax()
        best_gene_int = int(sub.loc[best_idx, "best_gene"])
        best_gene_bits = format(best_gene_int, f"0{bits[land]}b")
        best_accuracy = sub.loc[best_idx, "best_accuracy"]

        rows.append({
            "landscape": land,
            "runs": len(sub),
            "global_optimum_accuracy": global_fitness,
            "avg_accuracy":            sub["best_accuracy"].mean(),
            "std_accuracy":            sub["best_accuracy"].std(),
            "best_run_accuracy":       best_accuracy,
            "best_run_gene":           best_gene_bits,
            "gap_to_optimum":          global_fitness - best_accuracy,
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv("optimum_hit_summary.csv", index=False)

plot_grid_search()


#for exp in experiments:
    #plot_heatmap(exp)
    #plot_fitness_distribution(exp)
    #plot_diversity(exp)

