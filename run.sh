#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
EPSILON="0.1"

echo "========================================"
echo "  Bio-AI Project 3 — Full Pipeline"
echo "  Seed: $SEED   Epsilon: $EPSILON"
echo "========================================"

# ── Build ────────────────────────────────────────────────────────────────────
echo ""
echo "[1/7] Building C++ project..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tail -1
cmake --build build --target main -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" 2>&1 | tail -2
echo "  Build complete."

# ── Prepare output directories ───────────────────────────────────────────────
mkdir -p output figures

# ── Export landscape CSVs ─────────────────────────────────────────────────────
echo ""
echo "[2/7] Exporting landscape CSVs..."
./build/main data/01-breast-w_lr_F.h5  --export output/breast-w.csv
./build/main data/05-credit-a_rf_F.h5  --export output/credit-a.csv
./build/main data/08-letter-r_knn_F.h5 --export output/letter-r.csv
echo "  Landscape CSVs exported."

# ── Run NSGA-II on HDF5 datasets ─────────────────────────────────────────────
echo ""
echo "[3/7] Running NSGA-II on HDF5 datasets..."
echo ""

./build/main data/01-breast-w_lr_F.h5  --nsga2 --out output/breast-w  --epsilon "$EPSILON" --seed "$SEED"
echo ""
./build/main data/05-credit-a_rf_F.h5  --nsga2 --out output/credit-a  --epsilon "$EPSILON" --seed "$SEED"
echo ""
./build/main data/08-letter-r_knn_F.h5 --nsga2 --out output/letter-r  --epsilon "$EPSILON" --seed "$SEED"

# ── Run NSGA-II on triangle landscape ────────────────────────────────────────
echo ""
echo "[4/7] Running NSGA-II on Triangle landscape (N=16, m=1, s=4)..."
echo ""
./build/main --triangle --out output/triangle --epsilon "$EPSILON" --seed "$SEED"

# ── Generate NSGA-II static plots ────────────────────────────────────────────
echo ""
echo "[5/7] Generating NSGA-II plots..."
python3 plot_nsga2.py \
  --prefix output/breast-w output/credit-a output/letter-r output/triangle \
  --save figures

# ── Generate landscape visualization plots ────────────────────────────────────
echo ""
echo "[6/7] Generating landscape visualization plots..."
python3 plot_landscape.py output/breast-w.csv  --save figures
python3 plot_landscape.py output/credit-a.csv  --save figures
python3 plot_landscape.py output/letter-r.csv  --save figures
python3 plot_landscape.py --triangle --tri-n 16 --tri-m 1 --tri-s 4 --save figures

# ── Generate NSGA-II evolution animations ─────────────────────────────────────
echo ""
echo "[7/7] Generating NSGA-II evolution animations..."
python3 animate_nsga2.py --snapshots output/breast-w_snapshots.csv  --landscape output/breast-w.csv  --save figures
python3 animate_nsga2.py --snapshots output/credit-a_snapshots.csv  --landscape output/credit-a.csv  --save figures
python3 animate_nsga2.py --snapshots output/letter-r_snapshots.csv  --landscape output/letter-r.csv  --save figures
python3 animate_nsga2.py --snapshots output/triangle_snapshots.csv  --triangle --tri-n 16 --tri-m 1 --tri-s 4 --save figures

echo ""
echo "========================================"
echo "  All done!"
echo "  NSGA-II outputs:  output/"
echo "  Figures:          figures/"
echo "========================================"
echo ""
echo "Parameter summary (same for all 4 landscapes):"
echo "  pop_size    = 2 * N^2"
echo "  generations = 100"
echo "  crossover   = 0.9"
echo "  mutation    = 1/N (adaptive: boost x1.5 if diversity < 0.25, decay x0.95 if > 0.50)"
echo "  epsilon     = $EPSILON  (penalty available in landscape, not used by NSGA-II objectives)"
echo "  seed        = $SEED"
