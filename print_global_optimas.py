import pandas as pd

landscapes = [
    {"name": "Triangle",      "file": "data/triangle.csv",         "fit_col": 1},
    {"name": "Breast Cancer", "file": "data/landscape_breast.csv", "fit_col": 3},
    {"name": "Credit",        "file": "data/landscape_credit.csv", "fit_col": 3},
    {"name": "Letter",        "file": "data/landscape_letter.csv", "fit_col": 3},
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

landscapes = [
    {"name": "Zoo",        "file": "data/landscape_zoo.csv",       "fit_col": 3},
    {"name": "Hepatitis",  "file": "data/landscape_hepatitis.csv", "fit_col": 3},
]

for l in landscapes:
    df = pd.read_csv(l["file"])
    idx = df.iloc[:, 0]
    fit = df.iloc[:, l["fit_col"]]
    best_row = fit.idxmax()
    n_bits = 19 if l["name"] == "Hepatitis" else 16
    print(f"{l['name']}:")
    print(f"  Index: {int(idx[best_row])}")
    print(f"  Bitstring: {int(idx[best_row]):0{n_bits}b}")
    print(f"  Fitness: {fit[best_row]:.6f}")
    print()
