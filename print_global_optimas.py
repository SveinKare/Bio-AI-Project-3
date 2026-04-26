import pandas as pd

landscapes = [
    {"name": "Triangle",      "file": "triangle.csv",         "fit_col": 1},
    {"name": "Breast Cancer", "file": "landscape_breast.csv", "fit_col": 3},
    {"name": "Credit",        "file": "landscape_credit.csv", "fit_col": 3},
    {"name": "Letter",        "file": "landscape_letter.csv", "fit_col": 3},
]

for l in landscapes:
    df = pd.read_csv(l["file"])
    idx = df.iloc[:, 0]
    fit = df.iloc[:, l["fit_col"]]
    best_row = fit.idxmax()
    print(f"{l['name']}:")
    print(f"  Index: {int(idx[best_row])}")
    print(f"  Bitstring: {int(idx[best_row]):016b}")
    print(f"  Fitness: {fit[best_row]:.6f}")
    print()
