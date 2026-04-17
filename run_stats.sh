#!/usr/bin/env bash
# Step 5 — repeated runs for fair comparison: NSGA-II, PSO, and SGA on each of the
# four landscapes. Results are appended to output/stats/runs_raw.csv (one row per
# run × algorithm × landscape). aggregate_stats.py computes mean and standard
# deviation of scalar_fitness per (algorithm, landscape).
#
# Usage:
#   ./run_stats.sh [N_RUNS] [BASE_SEED] [EPSILON]
# Defaults: 30 runs, base seed 1_000_000, epsilon 0.1 (match main / run.sh).
#
# For ~100 runs as in the assignment example:
#   ./run_stats.sh 100
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

N_RUNS="${1:-30}"
BASE_SEED="${2:-1000000}"
EPSILON="${3:-0.1}"

STATS_DIR="$ROOT/output/stats"
RAW_CSV="$STATS_DIR/runs_raw.csv"
SUMMARY_CSV="$STATS_DIR/summary_by_algorithm_landscape.csv"
SUMMARY_MD="$STATS_DIR/summary.md"
README="$STATS_DIR/README_experiment.txt"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "=========================================="
echo "  Statistical runs (Step 5)"
echo "  Runs per (algorithm × landscape): $N_RUNS"
echo "  Seeds: \$((BASE_SEED + run_index)), run_index=0..$((N_RUNS - 1))"
echo "  Epsilon: $EPSILON"
echo "  Output:  $RAW_CSV"
echo "=========================================="

echo ""
echo "[1/4] Build..."
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --target main -j"$JOBS"

mkdir -p "$STATS_DIR"
rm -f "$RAW_CSV"

cat > "$README" << EOF
Experiment protocol (hyper-parameters aligned across algorithms)
================================================================
Epsilon (feature penalty weight):     $EPSILON
Seeds:                                 BASE_SEED + run_index  (run_index = 0 .. $((N_RUNS - 1)))
Runs per algorithm per landscape:      $N_RUNS

NSGA-II (BinaryNSGA2 in nsga2.cpp)
  - pop_size     = 2 * N^2
  - generations  = 100
  - crossover    = 0.9
  - mutation     = 1/N with adaptive diversity-based adjustment (see nsga2.cpp)
  - objectives   = (maximize accuracy, minimize time) on HDF5; triangle uses (accuracy, feature count)
  - Scalar row metric: best individual by accuracy, scalar_fitness = accuracy - (k/N)*epsilon

Binary PSO (pso.cpp)
  - swarm_size   = 2 * N^2
  - iterations   = 100
  - inertia w    = 0.9 -> 0.4 (linear)
  - c1, c2       = 2.0, 2.0
  - v_max        = 6.0
  - mutation       linear decay mut_start=1/N -> mut_end=0 (no diversity boost; see pso.cpp)
  - Scalar row metric: global-best scalar_fitness (accuracy - penalty)

Single-objective GA (single_ga.cpp)
  - pop          = 200, generations = 50
  - crossover    = 0.05, mutation = 0.05
  - elites = 2, tournament k = 3, 1 crossover point
  - Scalar row metric: best individual fitness (accuracy - penalty)

Comparison column
-----------------
All rows store scalar_fitness in runs_raw.csv for direct comparison (higher is better
for maximization of accuracy minus penalty).

Controlled parameters during the run
------------------------------------
Only the RNG seed changes between repeated runs (--seed). Epsilon and all population sizes
are fixed above. NSGA-II retains its internal adaptive mutation; PSO uses the scheduled
mutation rate only.

Multiple optima
---------------
NSGA-II returns a Pareto front (many trade-off solutions). For the table we take the
front member with highest accuracy (same rule as main binary). PSO/SGA report one best
scalar per run.
EOF

echo ""
echo "[2/4] Running $N_RUNS × 3 algorithms × 4 landscapes (lite mode, no large CSV exports)..."
echo ""

# Sequential runs only: all processes append to the same CSV (no parallel writes).
for ((run = 0; run < N_RUNS; run++)); do
  SEED=$((BASE_SEED + run))
  for case in breast credit letter triangle; do
    # --out basename becomes the CSV "case" column (see caseFromPrefix in main.cpp).
    PFX="$STATS_DIR/${case}"

    if [[ "$case" == "breast" ]]; then
      H5="data/01-breast-w_lr_F.h5"
      ./build/main "$H5" --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main "$H5" --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case breast --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
    elif [[ "$case" == "credit" ]]; then
      H5="data/05-credit-a_rf_F.h5"
      ./build/main "$H5" --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main "$H5" --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case credit --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
    elif [[ "$case" == "letter" ]]; then
      H5="data/08-letter-r_knn_F.h5"
      ./build/main "$H5" --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main "$H5" --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case letter --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
    else
      ./build/main --triangle --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --triangle --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case triangle --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
    fi
  done
  if (( (run + 1) % 5 == 0 )) || (( run + 1 == N_RUNS )); then
    echo "  completed run_index 0..$run"
  fi
done

echo ""
echo "[3/4] Aggregate (mean, std) -> $SUMMARY_CSV"
python3 "$ROOT/aggregate_stats.py" "$RAW_CSV" --out-csv "$SUMMARY_CSV" --out-md "$SUMMARY_MD"

echo ""
echo "[4/4] Done."
echo "  Raw rows:    $RAW_CSV"
echo "  Summary:     $SUMMARY_CSV"
echo "  Markdown:    $SUMMARY_MD"
echo "  Notes:       $README"
echo "=========================================="
