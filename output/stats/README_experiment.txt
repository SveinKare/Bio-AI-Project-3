Experiment protocol (hyper-parameters aligned across algorithms)
================================================================
Epsilon (feature penalty weight):     0.1
Seeds:                                 BASE_SEED + run_index  (run_index = 0 .. 99)
Runs per algorithm per landscape:      100

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
