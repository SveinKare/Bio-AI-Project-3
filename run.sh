#!/usr/bin/env bash
# Run NSGA-II, PSO, and SGA on all four landscapes; export landscape CSVs for
# 3D plots; record every best solution in output/best/all_best_solutions.csv;
# write figures (3D + summary landscapes, SGA evolution GIFs) under figures/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED="${1:-42}"
EPSILON="${2:-0.1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
FIG_DIR="$ROOT/figures"

echo "========================================"
echo "  Bio-AI Project 3 — pipeline"
echo "  Figures: $FIG_DIR"
echo "  Best:    output/best/all_best_solutions.csv"
echo "  Seed:    $SEED   Epsilon: $EPSILON"
echo "========================================"

echo ""
echo "[1/9] Configure & build..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
cmake --build build --target main -j"$JOBS"

mkdir -p output output/best output/sga "$FIG_DIR"
rm -f output/best/all_best_solutions.csv

echo ""
echo "[2/9] Export landscape CSVs (for 3D / summary plots)..."
./build/main data/01-breast-w_lr_F.h5  --export output/breast-w.csv
./build/main data/05-credit-a_rf_F.h5  --export output/credit-a.csv
./build/main data/08-letter-r_knn_F.h5 --export output/letter-r.csv

echo ""
echo "[3/9] NSGA-II (HDF5 ×3 + triangle)..."
./build/main data/01-breast-w_lr_F.h5  --nsga2 --out output/breast-w  --epsilon "$EPSILON" --seed "$SEED"
./build/main data/05-credit-a_rf_F.h5  --nsga2 --out output/credit-a  --epsilon "$EPSILON" --seed "$SEED"
./build/main data/08-letter-r_knn_F.h5 --nsga2 --out output/letter-r  --epsilon "$EPSILON" --seed "$SEED"
./build/main --triangle --out output/triangle --epsilon "$EPSILON" --seed "$SEED"

echo ""
echo "[4/9] Binary PSO (HDF5 ×3 + triangle)..."
./build/main data/01-breast-w_lr_F.h5  --pso --out output/pso-breast-w  --epsilon "$EPSILON" --seed "$SEED"
./build/main data/05-credit-a_rf_F.h5  --pso --out output/pso-credit-a  --epsilon "$EPSILON" --seed "$SEED"
./build/main data/08-letter-r_knn_F.h5 --pso --out output/pso-letter-r  --epsilon "$EPSILON" --seed "$SEED"
./build/main --triangle --pso --out output/pso-triangle --epsilon "$EPSILON" --seed "$SEED"

echo ""
echo "[5/9] Single-objective GA (output/sga/, anim frames)..."
./build/main --sga --epsilon "$EPSILON" --seed "$SEED"

echo ""
echo "[6/9] Landscape global optima (text) → output/best/landscape_global_optima.txt..."
( cd output/sga && python3 "$ROOT/print_global_optimas.py" ) > output/best/landscape_global_optima.txt

echo ""
echo "[7/9] Landscape plots (2×2 summary + 3D hinged map) → figures/..."
python3 plot_landscape.py output/breast-w.csv  --save "$FIG_DIR"
python3 plot_landscape.py output/credit-a.csv  --save "$FIG_DIR"
python3 plot_landscape.py output/letter-r.csv  --save "$FIG_DIR"
python3 plot_landscape.py --triangle --tri-n 16 --tri-m 1 --tri-s 4 --save "$FIG_DIR"

echo ""
echo "[8/9] Evolution GIFs (NSGA-II, PSO, SGA × 4 datasets) → figures/..."
python3 "$ROOT/animate.py" --figures-dir "$FIG_DIR"

echo ""
echo "[9/9] Done."
echo "  Plots & GIFs:  $FIG_DIR  (e.g. anim_nsga2_*.gif, anim_pso_*.gif, anim_sga_*.gif)"
echo "  Best runs:     output/best/all_best_solutions.csv"
echo "  Landscape max: output/best/landscape_global_optima.txt"
echo "  SGA bundle:     output/sga/ (stats, pop CSV, anim/*.csv)"
echo "  Algorithm logs: output/*.csv"
echo "========================================"
