#!/usr/bin/env python3
"""
Grid search over binary PSO hyperparameters by invoking the compiled `main` binary.

Requires CLI flags added in main.cpp: --pso-swarm, --pso-iters, --pso-w-start, etc.

Example:
  ./scripts/grid_search_pso.py \\
    --binary build/main \\
    --hdf5 data/01-breast-w_lr_F.h5 \\
    --out-dir output/pso_grid \\
    --lite

Override grids via JSON (lists of numbers), e.g. grid.json:
  {
    "swarm": [128, 256],
    "iters": [50, 100],
    "w_start": [0.7, 0.9],
    "w_end": [0.3, 0.4],
    "c1": [1.5, 2.0],
    "c2": [1.5, 2.0],
    "vmax": [6.0],
    "mut_start": [0.05],
    "mut_end": [0.0]
  }

Keys are optional; omitted keys are not passed (binary uses defaults for N).

Outputs (under --out-dir):
  grid_results.csv       — one row per run (every seed); includes config_index, seed
  grid_aggregate.csv     — one row per grid point: means, stdev, best fitness + seed
  grid_showcase.md       — Markdown summary (aggregate when --repeats > 1)
  grid_showcase_runs.md  — Markdown of raw runs (only if --write-per-run-md)

Use --repeats N to average over N seeds (seeds = --seed, --seed+1, …).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

# Default search space (edit or pass --grid-json). Omit keys for binary defaults.
# Cartesian product size multiplies quickly — use --grid-json or --max-runs.
DEFAULT_GRID: dict[str, list[Any]] = {
    "w_start": [0.7, 0.9],
    "w_end": [0.3, 0.4],
    "iters": [50, 100],
}

FLAG_KEYS = (
    ("swarm", "--pso-swarm"),
    ("iters", "--pso-iters"),
    ("w_start", "--pso-w-start"),
    ("w_end", "--pso-w-end"),
    ("c1", "--pso-c1"),
    ("c2", "--pso-c2"),
    ("vmax", "--pso-vmax"),
    ("mut_start", "--pso-mut-start"),
    ("mut_end", "--pso-mut-end"),
)

_DISPLAY_HEADERS: dict[str, str] = {
    "run_index": "Run",
    "config_index": "Config",
    "repeat_index": "Repeat",
    "seed": "Seed",
    "exit_code": "Exit",
    "swarm": "Swarm (grid)",
    "swarm_effective": "Swarm (actual)",
    "iters": "Iterations",
    "w_start": "w_start",
    "w_end": "w_end",
    "c1": "c₁",
    "c2": "c₂",
    "vmax": "v_max",
    "mut_start": "mut_start",
    "mut_end": "mut_end",
    "fitness": "Fitness",
    "accuracy": "Accuracy",
    "n_features": "# features",
    "time": "Time (obj₂)",
    "out_prefix": "Output prefix",
    "best_csv": "Best CSV",
}

_AGG_HEADERS: dict[str, str] = {
    "config_index": "Config",
    "n_ok": "OK runs",
    "n_fail": "Failed",
    "fitness_mean": "Fitness μ",
    "fitness_stdev": "Fitness σ",
    "fitness_max": "Fitness best",
    "accuracy_mean": "Accuracy μ",
    "accuracy_at_best_fit": "Accuracy @ best fit",
    "n_features_mean": "# feat μ",
    "time_mean": "Time μ",
    "best_seed": "Seed (best fit)",
    "best_run_index": "Run # (best)",
}


def _markdown_columns_runs(
    rows: list[dict[str, Any]],
    grid: dict[str, list[Any]],
) -> list[str]:
    """Per-seed table: ids + hyperparameters + metrics."""
    out: list[str] = [
        "run_index",
        "config_index",
        "repeat_index",
        "seed",
    ]
    for key, _ in FLAG_KEYS:
        if key not in grid or not grid[key]:
            continue
        if any(_nonempty(r.get(key)) for r in rows):
            out.append(key)
    for k in ("swarm_effective", "fitness", "accuracy", "n_features", "time"):
        if any(_nonempty(r.get(k)) for r in rows):
            out.append(k)
    out.append("exit_code")
    return out


def _nonempty(v: Any) -> bool:
    return v is not None and v != ""


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aggregate_seed_trials(
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    trials: rows with fitness, accuracy, n_features, time, seed, exit_code, run_index.
    Best = maximum fitness (scalar objective).
    """
    ok = [
        t
        for t in trials
        if t.get("exit_code") == 0 and _num(t.get("fitness")) is not None
    ]
    failed = len(trials) - len(ok)
    if not ok:
        return {
            "n_ok": 0,
            "n_fail": failed,
            "fitness_mean": "",
            "fitness_stdev": "",
            "fitness_max": "",
            "accuracy_mean": "",
            "accuracy_at_best_fit": "",
            "n_features_mean": "",
            "time_mean": "",
            "best_seed": "",
            "best_run_index": "",
        }

    fits_f = [float(_num(t["fitness"])) for t in ok]
    best_i = max(range(len(ok)), key=lambda i: fits_f[i])
    best_t = ok[best_i]

    accs = [_num(t.get("accuracy")) for t in ok]
    accs_n = [a for a in accs if a is not None]
    nfs = [_num(t.get("n_features")) for t in ok]
    nfs_n = [n for n in nfs if n is not None]
    times = [_num(t.get("time")) for t in ok]
    times_n = [x for x in times if x is not None]

    fit_mean = statistics.mean(fits_f)
    fit_std = statistics.stdev(fits_f) if len(fits_f) > 1 else 0.0

    return {
        "n_ok": len(ok),
        "n_fail": failed,
        "fitness_mean": fit_mean,
        "fitness_stdev": fit_std,
        "fitness_max": max(fits_f),
        "accuracy_mean": statistics.mean(accs_n) if accs_n else "",
        "accuracy_at_best_fit": best_t.get("accuracy", ""),
        "n_features_mean": statistics.mean(nfs_n) if nfs_n else "",
        "time_mean": statistics.mean(times_n) if times_n else "",
        "best_seed": best_t.get("seed", ""),
        "best_run_index": best_t.get("run_index", ""),
    }

_FITNESS_RE = re.compile(r"Fitness:\s+([\d.eE+-]+|inf|nan)")
_ACCURACY_RE = re.compile(r"Accuracy:\s+([\d.eE+-]+|inf|nan)")
_FEATURES_RE = re.compile(r"Features:\s+(\d+)\s*/\s*(\d+)")
_TIME_RE = re.compile(r"Time:\s+([\d.eE+-]+|inf|nan)")
_SWARM_RE = re.compile(r"swarm=(\d+)")


def parse_run_metrics(text: str) -> dict[str, Any]:
    """Values printed by the binary (best solution + run banner)."""
    out: dict[str, Any] = {
        "fitness": None,
        "accuracy": None,
        "n_features": None,
        "n_dims": None,
        "time": None,
        "swarm_effective": None,
    }
    for line in text.splitlines():
        if out["fitness"] is None:
            m = _FITNESS_RE.search(line)
            if m:
                out["fitness"] = float(m.group(1))
        if out["accuracy"] is None:
            m = _ACCURACY_RE.search(line)
            if m:
                out["accuracy"] = float(m.group(1))
        if out["n_features"] is None:
            m = _FEATURES_RE.search(line)
            if m:
                out["n_features"] = int(m.group(1))
                out["n_dims"] = int(m.group(2))
        if out["time"] is None:
            m = _TIME_RE.search(line)
            if m:
                out["time"] = float(m.group(1))
        if out["swarm_effective"] is None:
            m = _SWARM_RE.search(line)
            if m:
                out["swarm_effective"] = int(m.group(1))
    return out


def write_markdown_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    title: str,
    header_map: dict[str, str] | None = None,
) -> None:
    """GitHub-flavoured markdown table for reports / slides."""

    def cell(v: Any) -> str:
        if v is None or v == "":
            return "—"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return f"{v:.6g}"
        s = str(v).replace("\n", " ").strip()
        return s.replace("|", "\\|")

    hm = header_map or {}
    headers = [hm.get(c, c.replace("_", " ").title()) for c in columns]
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(c)) for c in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def iter_grid(
    grid: dict[str, list[Any]],
) -> Iterator[tuple[tuple[str, str, Any], ...]]:
    """Cartesian product of all non-empty keys in `grid`."""
    keys = [k for k, _ in FLAG_KEYS if k in grid and grid[k]]
    if not keys:
        yield ()
        return
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield tuple(zip(keys, combo))


def build_argv(
    base: list[str],
    combo: tuple[tuple[str, Any], ...],
) -> list[str]:
    out = list(base)
    for key, val in combo:
        flag = next(f for k, f in FLAG_KEYS if k == key)
        out += [flag, str(val)]
    return out


def combo_to_hyper_dict(combo: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    d: dict[str, Any] = {k: "" for k, _ in FLAG_KEYS}
    for key, val in combo:
        d[key] = val
    return d


def _markdown_columns_aggregate(
    grid: dict[str, list[Any]],
) -> list[str]:
    cols = ["config_index"]
    for key, _ in FLAG_KEYS:
        if key in grid and grid[key]:
            cols.append(key)
    cols += [
        "n_ok",
        "n_fail",
        "fitness_mean",
        "fitness_stdev",
        "fitness_max",
        "accuracy_mean",
        "accuracy_at_best_fit",
        "n_features_mean",
        "time_mean",
        "best_seed",
        "best_run_index",
    ]
    return cols


def main() -> int:
    p = argparse.ArgumentParser(description="PSO hyperparameter grid search")
    p.add_argument(
        "--binary",
        type=Path,
        default=Path("build/main"),
        help="Path to compiled main executable",
    )
    p.add_argument(
        "--hdf5",
        type=Path,
        default=Path("data/01-breast-w_lr_F.h5"),
        help="HDF5 landscape file",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/pso_grid"),
        help="Directory for CSV logs and per-run prefixes",
    )
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lite", action="store_true", help="Pass --lite to binary")
    p.add_argument(
        "--grid-json",
        type=Path,
        help="JSON file with grid lists (see module docstring)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Stop after this many runs (0 = no limit)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Append stdout/stderr tail column to grid_results.csv for debugging",
    )
    p.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing grid_showcase.md (aggregate summary)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help="Runs per grid point with seeds seed, seed+1, …, seed+N-1 (default: 1)",
    )
    p.add_argument(
        "--write-per-run-md",
        action="store_true",
        help="Also write grid_showcase_runs.md (one row per seed run)",
    )
    args = p.parse_args()

    grid: dict[str, list[Any]] = dict(DEFAULT_GRID)
    if args.grid_json:
        with open(args.grid_json, encoding="utf-8") as f:
            user = json.load(f)
        grid.update(user)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "grid_results.csv"
    aggregate_path = args.out_dir / "grid_aggregate.csv"
    md_path = args.out_dir / "grid_showcase.md"
    md_runs_path = args.out_dir / "grid_showcase_runs.md"

    repeats = max(1, args.repeats)

    base_cmd: list[str] = [
        str(args.binary.resolve()),
        str(args.hdf5),
        "--pso",
        "--epsilon",
        str(args.epsilon),
    ]
    if args.lite:
        base_cmd.append("--lite")

    compact_fields = [
        "run_index",
        "config_index",
        "repeat_index",
        "seed",
        "exit_code",
        "swarm",
        "swarm_effective",
        "iters",
        "w_start",
        "w_end",
        "c1",
        "c2",
        "vmax",
        "mut_start",
        "mut_end",
        "fitness",
        "accuracy",
        "n_features",
        "time",
        "out_prefix",
        "best_csv",
    ]
    fieldnames = list(compact_fields)
    if args.verbose:
        fieldnames.append("stdout_tail")

    combos = list(iter_grid(grid))
    if not combos:
        combos = [()]

    n_configs = len(combos)
    if args.max_runs > 0:
        n_configs = min(n_configs, args.max_runs)

    total_subprocess = n_configs * repeats
    done = 0

    showcase_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    agg_fieldnames = ["config_index"] + [
        k for k, _ in FLAG_KEYS
    ] + [
        "n_ok",
        "n_fail",
        "fitness_mean",
        "fitness_stdev",
        "fitness_max",
        "accuracy_mean",
        "accuracy_at_best_fit",
        "n_features_mean",
        "time_mean",
        "best_seed",
        "best_run_index",
    ]

    run_index = 0

    with open(results_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for config_index in range(n_configs):
            combo = combos[config_index]
            hyper = combo_to_hyper_dict(combo)
            trial_rows: list[dict[str, Any]] = []

            for repeat_index in range(repeats):
                seed = args.seed + repeat_index
                row: dict[str, Any] = {k: "" for k in compact_fields}
                row["run_index"] = run_index
                row["config_index"] = config_index
                row["repeat_index"] = repeat_index
                row["seed"] = seed
                for k, v in hyper.items():
                    row[k] = v

                prefix = args.out_dir / f"run_{run_index:04d}"
                best_csv = args.out_dir / f"best_run_{run_index:04d}.csv"
                cmd = (
                    base_cmd
                    + ["--seed", str(seed)]
                    + [
                        "--out",
                        str(prefix),
                        "--best-csv",
                        str(best_csv),
                        "--stats-run",
                        str(run_index),
                    ]
                )
                cmd = build_argv(cmd, combo)

                if args.dry_run:
                    print(" ".join(cmd))
                    run_index += 1
                    continue

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )
                text = proc.stdout + "\n" + proc.stderr
                metrics = parse_run_metrics(text)
                stdout_tail = text[-2000:] if len(text) > 2000 else text
                stdout_tail = stdout_tail.replace("\r", " ").strip()

                row["exit_code"] = proc.returncode
                row["swarm_effective"] = (
                    metrics["swarm_effective"]
                    if metrics["swarm_effective"] is not None
                    else ""
                )
                row["fitness"] = (
                    metrics["fitness"] if metrics["fitness"] is not None else ""
                )
                row["accuracy"] = (
                    metrics["accuracy"] if metrics["accuracy"] is not None else ""
                )
                row["n_features"] = (
                    metrics["n_features"]
                    if metrics["n_features"] is not None
                    else ""
                )
                row["time"] = (
                    metrics["time"] if metrics["time"] is not None else ""
                )
                row["out_prefix"] = str(prefix)
                row["best_csv"] = str(best_csv)
                if args.verbose:
                    row["stdout_tail"] = stdout_tail

                writer.writerow(row)
                csvfile.flush()
                showcase_rows.append(dict(row))
                trial_rows.append(dict(row))

                done += 1
                print(
                    f"[{done}/{total_subprocess}] cfg={config_index} "
                    f"seed={seed} exit={proc.returncode} "
                    f"fitness={metrics['fitness']} accuracy={metrics['accuracy']}",
                    file=sys.stderr,
                )
                run_index += 1

            if not args.dry_run and trial_rows:
                agg = aggregate_seed_trials(trial_rows)
                agg_row: dict[str, Any] = {
                    "config_index": config_index,
                    **hyper,
                    **agg,
                }
                aggregate_rows.append(agg_row)

    if not args.dry_run and aggregate_rows:
        with open(aggregate_path, "w", newline="", encoding="utf-8") as af:
            aw = csv.DictWriter(af, fieldnames=agg_fieldnames)
            aw.writeheader()
            for r in aggregate_rows:
                aw.writerow(r)
        print(f"Wrote {aggregate_path}", file=sys.stderr)

    merged_headers = {**_DISPLAY_HEADERS, **_AGG_HEADERS}

    if not args.dry_run and aggregate_rows and not args.no_markdown:
        md_cols = _markdown_columns_aggregate(grid)
        write_markdown_table(
            md_path,
            aggregate_rows,
            md_cols,
            "PSO grid search — mean and best over seeds",
            header_map=merged_headers,
        )
        print(f"Wrote {md_path}", file=sys.stderr)

    if (
        not args.dry_run
        and showcase_rows
        and args.write_per_run_md
        and not args.no_markdown
    ):
        md_cols_runs = _markdown_columns_runs(showcase_rows, grid)
        write_markdown_table(
            md_runs_path,
            showcase_rows,
            md_cols_runs,
            "PSO — individual runs (all seeds)",
            header_map=merged_headers,
        )
        print(f"Wrote {md_runs_path}", file=sys.stderr)

    if not args.dry_run:
        print(f"Wrote {results_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
