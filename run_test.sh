#!/usr/bin/env bash
# Run NSGA-II, PSO, and SGA on the test landscapes (zoo, hepatitis, asymmetric
# triangle); export landscape CSVs for plots; record every best solution in
# output/test/best/all_best_solutions.csv; write figures under figures/test/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SEED="${1:-42}"
EPSILON="${2:-0.1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
FIG_DIR="$ROOT/figures/test"

echo "========================================"
echo "  Bio-AI Project 3 — TEST pipeline"
echo "  Figures: $FIG_DIR"
echo "  Best:    output/test/best/all_best_solutions.csv"
echo "  Seed:    $SEED   Epsilon: $EPSILON"
echo "========================================"

echo ""
echo "[1/8] Configure & build..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
cmake --build build --target main -j"$JOBS"

mkdir -p output/test output/test/best output/test/sga "$FIG_DIR"
rm -f output/test/best/all_best_solutions.csv

echo ""
echo "[2/8] Export landscape CSVs (for plots)..."
./build/main data/test/06-zoo_lr_F.h5        --export output/test/zoo.csv
./build/main data/test/10-hepatitis_lr_F.h5   --export output/test/hepatitis.csv

BEST_CSV="output/test/best/all_best_solutions.csv"

echo ""
echo "[3/8] NSGA-II (HDF5 ×2 + asymmetric triangle)..."
./build/main data/test/06-zoo_lr_F.h5        --nsga2 --out output/test/zoo        --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"
./build/main data/test/10-hepatitis_lr_F.h5   --nsga2 --out output/test/hepatitis  --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"
./build/main --asym-triangle                  --out output/test/asym-triangle      --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"

echo ""
echo "[4/8] Binary PSO (HDF5 ×2 + asymmetric triangle)..."
./build/main data/test/06-zoo_lr_F.h5        --pso --out output/test/pso-zoo       --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"
./build/main data/test/10-hepatitis_lr_F.h5   --pso --out output/test/pso-hepatitis --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"
./build/main --asym-triangle --pso            --out output/test/pso-asym-triangle   --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"

echo ""
echo "[5/8] Single-objective GA (output/test/sga/)..."
./build/main --sga-test --epsilon "$EPSILON" --seed "$SEED" --best-csv "$BEST_CSV"

echo ""
echo "[6/8] Landscape plots (2×2 summary + 3D where feasible) → $FIG_DIR..."
python3 plot_landscape.py output/test/zoo.csv       --save "$FIG_DIR"
python3 plot_landscape.py output/test/hepatitis.csv --save "$FIG_DIR"
python3 plot_landscape.py --asym-triangle           --save "$FIG_DIR"

echo ""
echo "[7/8] Evolution GIFs (NSGA-II, PSO, SGA × zoo, hepatitis) → $FIG_DIR..."
python3 "$ROOT/animate.py" --test --figures-dir "$FIG_DIR"

echo ""
echo "[8/8] Done."
echo "  Plots & GIFs:  $FIG_DIR"
echo "  Best runs:     output/test/best/all_best_solutions.csv"
echo "  SGA bundle:    output/test/sga/"
echo "  Algorithm logs: output/test/*.csv"
echo "========================================"
