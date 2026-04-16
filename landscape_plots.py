import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

experiments = [
    {"name": "Triangle",       "landscape": "triangle.csv",         "bits": 16, "fit_col": 1},
    {"name": "Breast Cancer",  "landscape": "landscape_breast.csv", "bits": 9,  "fit_col": 3},
    {"name": "Credit",         "landscape": "landscape_credit.csv", "bits": 15, "fit_col": 3},
    {"name": "Letter",         "landscape": "landscape_letter.csv", "bits": 16, "fit_col": 3},
]

def find_true_optima(fitness, bits, top_n=50, tolerance=0.001):
    total = 1 << bits
    optima = []
    for i in range(total):
        f = fitness[i]
        if f == 0:
            continue
        is_optimum = True
        for bit in range(bits):
            neighbor = i ^ (1 << bit)
            if fitness[neighbor] > f + tolerance:
                is_optimum = False
                break
        if is_optimum:
            optima.append((i, f))
    optima.sort(key=lambda x: x[1], reverse=True)
    return optima[:top_n]

def plot_landscape(exp):
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

    # Global optimum
    global_idx = np.argmax(fitness)
    global_fit = fitness[global_idx]
    global_bits = format(global_idx, f'0{bits}b')
    print(f"{exp['name']}:")
    print(f"  Global optimum: index={global_idx}, bitstring={global_bits}, fitness={global_fit:.6f}")

    optima = find_true_optima(fitness, bits, top_n=50)
    print(f"  Found {len(optima)} local optima")
    ox = [idx & ((1 << lower) - 1) for idx, _ in optima]
    oy = [idx >> lower for idx, _ in optima]
    of = [f for _, f in optima]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="equal")
    plt.colorbar(im, ax=ax, label="Fitness")

    ax.scatter(ox, oy, c=of, cmap="hot", s=40,
               edgecolors="white", linewidths=0.8, zorder=5,
               label=f"Top {len(optima)} local optima")

    # Highlight global optimum
    gx = global_idx & ((1 << lower) - 1)
    gy = global_idx >> lower
    ax.scatter([gx], [gy], s=150, facecolors="none",
               edgecolors="cyan", linewidths=2, zorder=6,
               label=f"Global optimum ({global_fit:.4f})")

    ax.set_xlabel(f"Lower {lower} bits")
    ax.set_ylabel(f"Upper {upper} bits")
    ax.set_title(f"{exp['name']} Landscape (N={bits})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"landscape_{exp['name'].lower().replace(' ', '_')}.png", dpi=200)
    plt.show()

for exp in experiments:
    plot_landscape(exp)
