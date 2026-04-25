#include <bitset>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "landscapes.cpp"
#include "nsga2.cpp"
#include "pso.cpp"
#include "single_ga.cpp"

namespace {

// ── PSO hyper-parameter overrides ───────────────────────────────────────────

struct PsoHyperParams {
  std::optional<int> swarm_size;
  std::optional<int> iterations;
  std::optional<double> w_start;
  std::optional<double> w_end;
  std::optional<double> c1;
  std::optional<double> c2;
  std::optional<double> v_max;
  std::optional<double> mut_start;
  std::optional<double> mut_end;
};

template <std::size_t N>
typename BinaryPSO<N>::Config makePsoConfig(unsigned seed,
                                             const PsoHyperParams& h) {
  typename BinaryPSO<N>::Config cfg;
  cfg.seed = seed;
  if (h.swarm_size) cfg.swarm_size = *h.swarm_size;
  if (h.iterations) cfg.iterations = *h.iterations;
  if (h.w_start) cfg.w_start = *h.w_start;
  if (h.w_end) cfg.w_end = *h.w_end;
  if (h.c1) cfg.c1 = *h.c1;
  if (h.c2) cfg.c2 = *h.c2;
  if (h.v_max) cfg.v_max = *h.v_max;
  if (h.mut_start) cfg.mut_start = *h.mut_start;
  if (h.mut_end) cfg.mut_end = *h.mut_end;
  return cfg;
}

// ── Output helpers ──────────────────────────────────────────────────────────

template <std::size_t N>
void printBestNSGA2(const std::vector<Individual<N>>& front) {
  if (front.empty()) return;

  const auto& best = *std::max_element(front.begin(), front.end(),
      [](const auto& a, const auto& b) {
        return a.obj_accuracy < b.obj_accuracy;
      });

  std::cout << "\n  -- Best solution found --\n"
            << "  Bitstring: " << best.chromosome.to_string() << '\n'
            << "  Features:  " << best.chromosome.count() << " / " << N << '\n'
            << "  Accuracy:  " << best.obj_accuracy << '\n'
            << "  Time:      " << best.obj_time << '\n';
}

template <std::size_t N>
void printBestPSO(const Particle<N>& best) {
  std::cout << "\n  -- Best solution found --\n"
            << "  Bitstring: " << best.position.to_string() << '\n'
            << "  Features:  " << best.position.count() << " / " << N << '\n'
            << "  Fitness:   " << best.fitness << '\n'
            << "  Accuracy:  " << best.obj_accuracy << '\n'
            << "  Time:      " << best.obj_time << '\n';
}

template <typename Algo>
void exportCSVs(Algo& algo, const std::string& prefix) {
  algo.exportParetoCSV(prefix + "_pareto.csv");
  algo.exportGenerationsCSV(prefix + "_gens.csv");
  algo.exportPopulationCSV(prefix + "_pop.csv");
  algo.exportSnapshotsCSV(prefix + "_snapshots.csv");

  std::cout << "\n  Exported: " << prefix << "_pareto.csv\n"
            << "  Exported: " << prefix << "_gens.csv\n"
            << "  Exported: " << prefix << "_pop.csv\n"
            << "  Exported: " << prefix << "_snapshots.csv\n";
}

std::string caseFromPrefix(const std::string& prefix) {
  const auto pos = prefix.find_last_of('/');
  if (pos == std::string::npos || pos + 1 >= prefix.size()) return prefix;
  return prefix.substr(pos + 1);
}

// ── Best-solution tracking (global state) ───────────────────────────────────

std::string g_best_csv_path;
int g_stats_run_index = -1;
unsigned g_stats_seed = 0;

void appendBestSolutionRow(const std::string& algorithm,
                           const std::string& case_id,
                           const std::string& bitstring,
                           std::size_t n_features,
                           double accuracy,
                           double second_objective,
                           double scalar_fitness) {
  const std::string& path = g_best_csv_path;
  {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
  }
  const bool need_header = !std::filesystem::exists(path) ||
      std::filesystem::file_size(path) == 0;
  std::ofstream out(path, std::ios::app);
  if (need_header) {
    if (g_stats_run_index >= 0) {
      out << "run_index,seed,algorithm,case,bitstring,n_features,accuracy,"
             "second_objective,scalar_fitness\n";
    } else {
      out << "algorithm,case,bitstring,n_features,accuracy,second_objective,"
             "scalar_fitness\n";
    }
  }
  if (g_stats_run_index >= 0) {
    out << g_stats_run_index << ',' << g_stats_seed << ',';
  }
  out << algorithm << ',' << case_id << ',' << bitstring << ','
      << n_features << ',' << accuracy << ',' << second_objective << ','
      << scalar_fitness << '\n';
}

template <std::size_t N>
void recordBestNSGA2(const std::vector<Individual<N>>& front,
                     const std::string& out_prefix,
                     double epsilon) {
  if (front.empty()) return;
  const auto& best = *std::max_element(front.begin(), front.end(),
      [](const auto& a, const auto& b) {
        return a.obj_accuracy < b.obj_accuracy;
      });
  const double pen =
      (static_cast<double>(best.chromosome.count()) / static_cast<double>(N)) *
      epsilon;
  const double scalar = best.obj_accuracy - pen;
  appendBestSolutionRow("nsga2", caseFromPrefix(out_prefix),
                        best.chromosome.to_string(), best.chromosome.count(),
                        best.obj_accuracy, best.obj_time, scalar);
}

template <std::size_t N>
void recordBestPSO(const Particle<N>& best, const std::string& out_prefix) {
  appendBestSolutionRow("pso", caseFromPrefix(out_prefix),
                        best.position.to_string(), best.position.count(),
                        best.obj_accuracy, best.obj_time, best.fitness);
}

// ── PSO runners ─────────────────────────────────────────────────────────────

template <std::size_t N>
void runPSO_HDF5(const std::string& hdf5_path,
                 const std::string& out_prefix,
                 double epsilon, unsigned seed,
                 const PsoHyperParams& hyper,
                 bool lite) {
  HDF5Landscape<N> landscape(hdf5_path, epsilon, true);

  typename BinaryPSO<N>::Config cfg = makePsoConfig<N>(seed, hyper);

  BinaryPSO<N> pso(
      [&landscape](const std::bitset<N>& chr) -> PSOEvalResult {
        auto f = landscape.fitness(chr);
        return {f.accuracy - f.penalty, f.accuracy, f.mean_time};
      },
      cfg);

  std::cout << "Running Binary PSO on " << hdf5_path
            << " (N=" << N
            << ", swarm=" << cfg.swarm_size
            << ", iters=" << cfg.iterations
            << ", w=" << cfg.w_start << "->" << cfg.w_end
            << ", c1=" << cfg.c1 << ", c2=" << cfg.c2
            << ", seed=" << seed << ")\n";

  pso.run();
  printBestPSO<N>(pso.globalBest());
  if (!lite) exportCSVs(pso, out_prefix);
  recordBestPSO<N>(pso.globalBest(), out_prefix);
}

template <std::size_t N>
void runPSO_Triangle(const std::string& out_prefix,
                     int m, int s,
                     double epsilon, unsigned seed,
                     const PsoHyperParams& hyper,
                     bool lite) {
  TriangleLandscape<N> landscape(m, s, epsilon);

  typename BinaryPSO<N>::Config cfg = makePsoConfig<N>(seed, hyper);

  BinaryPSO<N> pso(
      [&landscape](const std::bitset<N>& chr) -> PSOEvalResult {
        auto f = landscape.fitness(chr);
        return {f.accuracy - f.penalty, f.accuracy,
                static_cast<double>(chr.count())};
      },
      cfg);

  std::cout << "Running Binary PSO on Triangle landscape"
            << " (N=" << N << ", m=" << m << ", s=" << s
            << ", swarm=" << cfg.swarm_size
            << ", iters=" << cfg.iterations
            << ", w=" << cfg.w_start << "->" << cfg.w_end
            << ", seed=" << seed << ")\n";

  pso.run();
  printBestPSO<N>(pso.globalBest());
  if (!lite) exportCSVs(pso, out_prefix);
  recordBestPSO<N>(pso.globalBest(), out_prefix);
}

template <std::size_t N>
void runPSO_AsymTriangle(const std::string& out_prefix,
                         double epsilon, unsigned seed,
                         const PsoHyperParams& hyper,
                         bool lite) {
  AsymmetricTriangleLandscape<N> landscape(
      AsymmetricTriangleLandscape<N>::testTriangleFitness(), epsilon);

  typename BinaryPSO<N>::Config cfg = makePsoConfig<N>(seed, hyper);

  BinaryPSO<N> pso(
      [&landscape](const std::bitset<N>& chr) -> PSOEvalResult {
        auto f = landscape.fitness(chr);
        return {f.accuracy - f.penalty, f.accuracy,
                static_cast<double>(chr.count())};
      },
      cfg);

  std::cout << "Running Binary PSO on Asymmetric Triangle landscape"
            << " (N=" << N
            << ", swarm=" << cfg.swarm_size
            << ", iters=" << cfg.iterations
            << ", w=" << cfg.w_start << "->" << cfg.w_end
            << ", seed=" << seed << ")\n";

  pso.run();
  printBestPSO<N>(pso.globalBest());
  if (!lite) exportCSVs(pso, out_prefix);
  recordBestPSO<N>(pso.globalBest(), out_prefix);
}

// ── NSGA-II runners ─────────────────────────────────────────────────────────

template <std::size_t N>
void runNSGA2_HDF5(const std::string& hdf5_path,
                   const std::string& out_prefix,
                   double epsilon, unsigned seed,
                   bool lite) {
  HDF5Landscape<N> landscape(hdf5_path, epsilon, true);

  typename NSGA2<N>::Config cfg;
  cfg.seed = seed;
  cfg.maximize_first = true;
  cfg.minimize_second = true;

  NSGA2<N> nsga(
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
        auto f = landscape.fitness(chr);
        return {f.accuracy, f.mean_time};
      },
      cfg);

  std::cout << "Running NSGA-II on " << hdf5_path
            << " (N=" << N
            << ", pop=" << cfg.pop_size
            << ", gens=" << cfg.generations
            << ", mutation=" << cfg.mutation_rate
            << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';
  printBestNSGA2<N>(front);
  if (!lite) exportCSVs(nsga, out_prefix);
  recordBestNSGA2<N>(front, out_prefix, epsilon);
}

template <std::size_t N>
void runNSGA2_Triangle(const std::string& out_prefix,
                       int m, int s,
                       double epsilon, unsigned seed,
                       bool lite) {
  TriangleLandscape<N> landscape(m, s, epsilon);

  typename NSGA2<N>::Config cfg;
  cfg.seed = seed;
  cfg.maximize_first = true;
  cfg.minimize_second = true;

  NSGA2<N> nsga(
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
        auto f = landscape.fitness(chr);
        return {f.accuracy, static_cast<double>(chr.count())};
      },
      cfg);

  std::cout << "Running NSGA-II on Triangle landscape"
            << " (N=" << N << ", m=" << m << ", s=" << s
            << ", pop=" << cfg.pop_size
            << ", gens=" << cfg.generations
            << ", mutation=" << cfg.mutation_rate
            << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';
  printBestNSGA2<N>(front);
  if (!lite) exportCSVs(nsga, out_prefix);
  recordBestNSGA2<N>(front, out_prefix, epsilon);
}

template <std::size_t N>
void runNSGA2_AsymTriangle(const std::string& out_prefix,
                           double epsilon, unsigned seed,
                           bool lite) {
  AsymmetricTriangleLandscape<N> landscape(
      AsymmetricTriangleLandscape<N>::testTriangleFitness(), epsilon);

  typename NSGA2<N>::Config cfg;
  cfg.seed = seed;
  cfg.maximize_first = true;
  cfg.minimize_second = true;

  NSGA2<N> nsga(
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
        auto f = landscape.fitness(chr);
        return {f.accuracy, static_cast<double>(chr.count())};
      },
      cfg);

  std::cout << "Running NSGA-II on Asymmetric Triangle landscape"
            << " (N=" << N
            << ", pop=" << cfg.pop_size
            << ", gens=" << cfg.generations
            << ", mutation=" << cfg.mutation_rate
            << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';
  printBestNSGA2<N>(front);
  if (!lite) exportCSVs(nsga, out_prefix);
  recordBestNSGA2<N>(front, out_prefix, epsilon);
}

// ── SGA runner ──────────────────────────────────────────────────────────────

template <std::size_t N>
void runSGAExperiment(std::unique_ptr<Landscape<N>> landscape,
                      const std::string& name,
                      bool lite,
                      const std::string& sga_dir) {
  SingleObjectiveGA<N> ga(
      std::move(landscape),
      200,   // popsize
      50,    // generations
      0.05,  // crossover
      0.05,  // mutation
      2,     // elites
      3,     // kParents
      1,     // crossover points
      name,
      sga_dir);
  ga.init();
  auto res = ga.run();
  std::cout << name << " - Accuracy: " << res.getAccuracy() << " Bitstring: "
            << res.getGene() << std::endl;

  if (!lite) {
    std::ofstream out(sga_dir + "/pop_" + name + ".csv");
    out << "id,fitness,niche" << std::endl;
    auto niches = ga.findNiches();
    for (size_t i = 0; i < niches.size(); i++) {
      for (auto& ind : niches[i]) {
        out << ind.getGene().to_ulong() << "," << ind.getAccuracy() << "," << i
            << std::endl;
      }
    }
  }

  const double pen = res.getAccuracy() - res.getFitness();
  appendBestSolutionRow("sga", name, res.getGene().to_string(),
                        res.getGene().count(), res.getAccuracy(), pen,
                        res.getFitness());
}

// ── Training phase ──────────────────────────────────────────────────────────
// All 3 algorithms × 4 datasets (breast, credit, letter, triangle)
// repeated n_runs times with incrementing seeds.

void runTraining(int n_runs, unsigned base_seed, double epsilon) {
  const std::string dir = "results/train";
  const std::string sga_dir = dir + "/sga";
  std::filesystem::create_directories(dir);
  std::filesystem::create_directories(sga_dir);

  g_best_csv_path = dir + "/runs_raw.csv";
  std::filesystem::remove(g_best_csv_path);

  std::cout << "==========================================\n"
            << "  TRAINING PHASE\n"
            << "  Runs: " << n_runs
            << "  Base seed: " << base_seed
            << "  Epsilon: " << epsilon << '\n'
            << "  Algorithms: NSGA-II, PSO, SGA\n"
            << "  Datasets: breast (N=9), credit (N=15),\n"
            << "            letter (N=16), triangle (N=16)\n"
            << "  Output: " << g_best_csv_path << '\n'
            << "==========================================\n\n";

  const PsoHyperParams pso_defaults;

  for (int run = 0; run < n_runs; ++run) {
    unsigned seed = base_seed + static_cast<unsigned>(run);
    g_stats_run_index = run;
    g_stats_seed = seed;

    // ── NSGA-II ──
    runNSGA2_HDF5<9>("data/01-breast-w_lr_F.h5", dir + "/breast",
                     epsilon, seed, true);
    runNSGA2_HDF5<15>("data/05-credit-a_rf_F.h5", dir + "/credit",
                      epsilon, seed, true);
    runNSGA2_HDF5<16>("data/08-letter-r_knn_F.h5", dir + "/letter",
                      epsilon, seed, true);
    runNSGA2_Triangle<16>(dir + "/triangle", 1, 4, epsilon, seed, true);

    // ── PSO ──
    runPSO_HDF5<9>("data/01-breast-w_lr_F.h5", dir + "/breast",
                   epsilon, seed, pso_defaults, true);
    runPSO_HDF5<15>("data/05-credit-a_rf_F.h5", dir + "/credit",
                    epsilon, seed, pso_defaults, true);
    runPSO_HDF5<16>("data/08-letter-r_knn_F.h5", dir + "/letter",
                    epsilon, seed, pso_defaults, true);
    runPSO_Triangle<16>(dir + "/triangle", 1, 4, epsilon, seed,
                        pso_defaults, true);

    // ── SGA ──
    seedSgaRng(seed);
    {
      auto ls = std::make_unique<HDF5Landscape<9>>(
          "data/01-breast-w_lr_F.h5", epsilon);
      runSGAExperiment<9>(std::move(ls), "breast", true, sga_dir);
    }
    {
      auto ls = std::make_unique<HDF5Landscape<15>>(
          "data/05-credit-a_rf_F.h5", epsilon);
      runSGAExperiment<15>(std::move(ls), "credit", true, sga_dir);
    }
    {
      auto ls = std::make_unique<HDF5Landscape<16>>(
          "data/08-letter-r_knn_F.h5", epsilon);
      runSGAExperiment<16>(std::move(ls), "letter", true, sga_dir);
    }
    {
      auto ls = std::make_unique<TriangleLandscape<16>>(1, 4, epsilon);
      runSGAExperiment<16>(std::move(ls), "triangle", true, sga_dir);
    }

    if ((run + 1) % 5 == 0 || run + 1 == n_runs) {
      std::cout << "\n  >>> Completed run " << (run + 1)
                << " / " << n_runs << " <<<\n\n";
    }
  }

  std::cout << "Training complete. Raw results: " << g_best_csv_path << '\n';
}

// ── Test phase ──────────────────────────────────────────────────────────────
// NSGA-II only on zoo (N=16), hepatitis (N=19), asymmetric triangle (N=31).

void runTest(unsigned seed, double epsilon) {
  const std::string dir = "results/test";
  std::filesystem::create_directories(dir);

  g_best_csv_path = dir + "/best_solutions.csv";
  std::filesystem::remove(g_best_csv_path);
  g_stats_run_index = -1;
  g_stats_seed = seed;

  std::cout << "==========================================\n"
            << "  TEST PHASE\n"
            << "  Seed: " << seed
            << "  Epsilon: " << epsilon << '\n'
            << "  Algorithm: NSGA-II\n"
            << "  Datasets: zoo (N=16), hepatitis (N=19),\n"
            << "            asym-triangle (N=31)\n"
            << "  Output: " << dir << "/\n"
            << "==========================================\n\n";

  runNSGA2_HDF5<16>("data/test/06-zoo_lr_F.h5",
                    dir + "/zoo", epsilon, seed, false);
  runNSGA2_HDF5<19>("data/test/10-hepatitis_lr_F.h5",
                    dir + "/hepatitis", epsilon, seed, false);
  runNSGA2_AsymTriangle<31>(dir + "/asym-triangle", epsilon, seed, false);

  std::cout << "\nTest complete. Results saved to: " << dir << "/\n";
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
  bool train_only = false;
  bool test_only = false;
  int n_runs = 30;
  unsigned seed = 42;
  double epsilon = 0.1;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--train") {
      args.train_only = true;
    } else if (arg == "--test") {
      args.test_only = true;
    } else if (arg == "--runs" && i + 1 < argc) {
      args.n_runs = std::stoi(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      args.seed = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg == "--epsilon" && i + 1 < argc) {
      args.epsilon = std::stod(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: main [OPTIONS]\n\n"
                << "No flags:  run both training and test phases\n\n"
                << "Options:\n"
                << "  --train          Run only the training phase\n"
                << "  --test           Run only the test phase\n"
                << "  --runs N         Number of training runs (default: 30)\n"
                << "  --seed N         Base seed (default: 42)\n"
                << "  --epsilon F      Feature penalty weight (default: 0.1)\n"
                << "  -h, --help       Show this help\n";
      std::exit(0);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);

    const bool run_train = !args.test_only;
    const bool run_test = !args.train_only;

    if (run_train)
      runTraining(args.n_runs, args.seed, args.epsilon);

    if (run_test)
      runTest(args.seed, args.epsilon);

  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }

  return 0;
}
