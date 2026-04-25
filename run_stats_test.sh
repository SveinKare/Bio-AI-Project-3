#!/usr/bin/env bash
# Repeated runs for fair comparison on TEST landscapes: NSGA-II, PSO, and SGA
# on zoo (N=16), hepatitis (N=19), and asymmetric triangle (N=31).
#
# Usage:
#   ./run_stats_test.sh [N_RUNS] [BASE_SEED] [EPSILON]
# Defaults: 30 runs, base seed 2_000_000, epsilon 0.1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

N_RUNS="${1:-30}"
BASE_SEED="${2:-2000000}"
EPSILON="${3:-0.1}"

STATS_DIR="$ROOT/output/test/stats"
RAW_CSV="$STATS_DIR/runs_raw.csv"
SUMMARY_CSV="$STATS_DIR/summary_by_algorithm_landscape.csv"
SUMMARY_MD="$STATS_DIR/summary.md"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

echo "=========================================="
echo "  Statistical runs — TEST landscapes"
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

echo ""
echo "[2/4] Running $N_RUNS × 3 algorithms × 3 test landscapes (lite mode)..."
echo ""

for ((run = 0; run < N_RUNS; run++)); do
  SEED=$((BASE_SEED + run))
  for case in zoo hepatitis asym-triangle; do
    PFX="$STATS_DIR/${case}"

    if [[ "$case" == "zoo" ]]; then
      H5="data/test/06-zoo_lr_F.h5"
      ./build/main "$H5" --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main "$H5" --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case zoo --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
    elif [[ "$case" == "hepatitis" ]]; then
      H5="data/test/10-hepatitis_lr_F.h5"
      ./build/main "$H5" --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main "$H5" --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case hepatitis --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
    else
      ./build/main --asym-triangle --nsga2 --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --asym-triangle --pso --out "$PFX" --epsilon "$EPSILON" --seed "$SEED" \
        --lite --best-csv "$RAW_CSV" --stats-run "$run"
      ./build/main --sga --sga-case asym-triangle --epsilon "$EPSILON" --seed "$SEED" \
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
echo "=========================================="
