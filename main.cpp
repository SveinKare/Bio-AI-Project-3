#include <bitset>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <memory>
#include <string>
#include <vector>

#include "landscapes.cpp"
#include "nsga2.cpp"
#include "pso.cpp"
#include "single_ga.cpp"
#include <memory>
#include <numeric>
#include <iomanip>

namespace {

bool isPowerOfTwo(std::size_t x) { return x != 0 && (x & (x - 1)) == 0; }

std::size_t inferFeatureCountFromFile(
    const std::string& hdf5_path,
    bool is_feature_selection_landscape) {
  H5::H5File file(hdf5_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet("accuracies");
  H5::DataSpace dataspace = dataset.getSpace();

  if (dataspace.getSimpleExtentNdims() != 2) {
    throw std::runtime_error("Dataset 'accuracies' must be 2D.");
  }
  file.close();

  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims, nullptr);
  const std::vector<std::size_t> candidates = {
      static_cast<std::size_t>(dims[0]), static_cast<std::size_t>(dims[1])};

  for (std::size_t dim : candidates) {
    const std::size_t target = is_feature_selection_landscape ? (dim + 1) : dim;
    if (!isPowerOfTwo(target)) continue;

    std::size_t n = 0;
    while ((std::size_t{1} << n) < target) ++n;
    if ((std::size_t{1} << n) == target) return n;
  }

  throw std::runtime_error(
      "Could not infer feature count from 'accuracies' shape.");
}

template <std::size_t N>
void runLandscape(const std::string& hdf5_path, const std::string& csv_path,
                  double epsilon) {
  HDF5Landscape<N> landscape(hdf5_path, epsilon, true);

  const std::bitset<N> only_feature_0(1UL);
  const std::bitset<N> all_features((std::size_t{1} << N) - 1);

  auto f1 = landscape.fitness(only_feature_0);
  auto f_all = landscape.fitness(all_features);

  std::cout << "Loaded: " << hdf5_path << '\n';
  std::cout << "Detected features: " << N << '\n';
  std::cout << "Mean accuracy for index 1: " << f1.accuracy << '\n';
  std::cout << "Mean time for index 1: " << f1.mean_time << '\n';
  std::cout << "Mean accuracy for all features: " << f_all.accuracy << '\n';

  if (!csv_path.empty()) {
    landscape.exportToCSV(csv_path);
    std::cout << "Exported CSV to: " << csv_path << '\n';
  }
}

// ── Shared helpers ──────────────────────────────────────────────────────────

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

constexpr const char* kBestSolutionsPath = "output/best/all_best_solutions.csv";

std::string g_best_csv_path = kBestSolutionsPath;
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
                 bool lite) {
  HDF5Landscape<N> landscape(hdf5_path, epsilon, true);

  typename BinaryPSO<N>::Config cfg;
  cfg.seed = seed;

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
                     bool lite) {
  TriangleLandscape<N> landscape(m, s, epsilon);

  typename BinaryPSO<N>::Config cfg;
  cfg.seed = seed;

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

// ── Single-objective GA (single-ga branch workflow / plotting outputs) ────

template <std::size_t N>
void runExperiment(std::unique_ptr<Landscape<N>> landscape,
                   const std::string& name,
                   bool lite) {
  constexpr const char* kSgaDir = "output/sga";
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
      kSgaDir);
  ga.init();
  auto res = ga.run();
  std::cout << name << " - Accuracy: " << res.getAccuracy() << " Bitstring: "
            << res.getGene() << std::endl;

  if (!lite) {
    std::ofstream out(std::string(kSgaDir) + "/pop_" + name + ".csv");
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

void runSingleGAExperiments(double epsilon, bool lite, unsigned seed) {
  seedSgaRng(seed);
  std::filesystem::create_directories("output/sga");

  {
    auto landscape = std::make_unique<TriangleLandscape<16>>(1, 4, epsilon);
    landscape->precompute("data/triangle.csv");
    runExperiment<16>(std::move(landscape), "triangle", lite);
    if (!lite) {
      std::error_code ec;
      std::filesystem::copy_file("data/triangle.csv", "output/sga/triangle.csv",
                                 std::filesystem::copy_options::overwrite_existing,
                                 ec);
    }
  }

  {
    auto landscape =
        std::make_unique<HDF5Landscape<9>>("data/01-breast-w_lr_F.h5", epsilon);
    if (!lite) landscape->exportToCSV("output/sga/landscape_breast.csv");
    runExperiment<9>(std::move(landscape), "breast", lite);
  }

  {
    auto landscape =
        std::make_unique<HDF5Landscape<15>>("data/05-credit-a_rf_F.h5", epsilon);
    if (!lite) landscape->exportToCSV("output/sga/landscape_credit.csv");
    runExperiment<15>(std::move(landscape), "credit", lite);
  }

  {
    auto landscape =
        std::make_unique<HDF5Landscape<16>>("data/08-letter-r_knn_F.h5", epsilon);
    if (!lite) landscape->exportToCSV("output/sga/landscape_letter.csv");
    runExperiment<16>(std::move(landscape), "letter", lite);
  }
}

void runSingleGACase(double epsilon, unsigned seed, const std::string& case_name,
                     bool lite) {
  seedSgaRng(seed);
  std::filesystem::create_directories("output/sga");
  if (case_name == "triangle") {
    auto landscape = std::make_unique<TriangleLandscape<16>>(1, 4, epsilon);
    landscape->precompute("data/triangle.csv");
    runExperiment<16>(std::move(landscape), "triangle", lite);
    return;
  }
  if (case_name == "breast") {
    auto landscape =
        std::make_unique<HDF5Landscape<9>>("data/01-breast-w_lr_F.h5", epsilon);
    if (!lite) landscape->exportToCSV("output/sga/landscape_breast.csv");
    runExperiment<9>(std::move(landscape), "breast", lite);
    return;
  }
  if (case_name == "credit") {
    auto landscape =
        std::make_unique<HDF5Landscape<15>>("data/05-credit-a_rf_F.h5", epsilon);
    if (!lite) landscape->exportToCSV("output/sga/landscape_credit.csv");
    runExperiment<15>(std::move(landscape), "credit", lite);
    return;
  }
  if (case_name == "letter") {
    auto landscape =
        std::make_unique<HDF5Landscape<16>>("data/08-letter-r_knn_F.h5", epsilon);
    if (!lite) landscape->exportToCSV("output/sga/landscape_letter.csv");
    runExperiment<16>(std::move(landscape), "letter", lite);
    return;
  }
  throw std::runtime_error("Unknown --sga-case: " + case_name);
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
  std::string hdf5_path = "data/01-breast-w_lr_F.h5";
  std::string csv_path;
  std::string out_prefix;
  std::string best_csv;
  std::string sga_case = "all";
  bool run_nsga2 = false;
  bool run_pso = false;
  bool run_triangle = false;
  bool run_sga = false;
  bool lite = false;
  double epsilon = 0.1;
  unsigned seed = 42;
  int stats_run = -1;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--export" && i + 1 < argc) {
      args.csv_path = argv[++i];
    } else if (arg == "--nsga2") {
      args.run_nsga2 = true;
    } else if (arg == "--pso") {
      args.run_pso = true;
    } else if (arg == "--triangle") {
      args.run_triangle = true;
    } else if (arg == "--sga") {
      args.run_sga = true;
    } else if (arg == "--lite") {
      args.lite = true;
    } else if (arg == "--best-csv" && i + 1 < argc) {
      args.best_csv = argv[++i];
    } else if (arg == "--sga-case" && i + 1 < argc) {
      args.sga_case = argv[++i];
    } else if (arg == "--stats-run" && i + 1 < argc) {
      args.stats_run = std::stoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      args.out_prefix = argv[++i];
    } else if (arg == "--epsilon" && i + 1 < argc) {
      args.epsilon = std::stod(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      args.seed = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (arg[0] != '-') {
      args.hdf5_path = arg;
    }
  }
  return args;
}

template <std::size_t N>
void dispatch(const Args& args) {
  const std::string& prefix = args.out_prefix;
  const bool lite = args.lite;
  const double eps = args.epsilon;
  if (args.run_triangle) {
    if (args.run_pso)
      runPSO_Triangle<N>(prefix.empty() ? "output/triangle" : prefix,
                         1, 4, eps, args.seed, lite);
    else
      runNSGA2_Triangle<N>(prefix.empty() ? "output/triangle" : prefix,
                           1, 4, eps, args.seed, lite);
  } else if (args.run_pso) {
    runPSO_HDF5<N>(args.hdf5_path,
                   prefix.empty() ? "output/pso" : prefix,
                   eps, args.seed, lite);
  } else if (args.run_nsga2) {
    runNSGA2_HDF5<N>(args.hdf5_path,
                     prefix.empty() ? "output/nsga2" : prefix,
                     eps, args.seed, lite);
  } else {
    runLandscape<N>(args.hdf5_path, args.csv_path, eps);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);

    if (!args.best_csv.empty()) g_best_csv_path = args.best_csv;
    g_stats_run_index = args.stats_run;
    g_stats_seed = args.seed;

    if (args.run_sga) {
      if (!args.sga_case.empty() && args.sga_case != "all") {
        runSingleGACase(args.epsilon, args.seed, args.sga_case, args.lite);
      } else {
        runSingleGAExperiments(args.epsilon, args.lite, args.seed);
      }
      return 0;
    }

    if (args.run_triangle) {
      dispatch<16>(args);
      return 0;
    }

    const std::size_t feature_count =
        inferFeatureCountFromFile(args.hdf5_path, true);

    switch (feature_count) {
      case 9:  dispatch<9>(args);  break;
      case 15: dispatch<15>(args); break;
      case 16: dispatch<16>(args); break;
      default:
        throw std::runtime_error(
            "Unsupported feature count " + std::to_string(feature_count));
    }
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
<<<<<<< HEAD
  }
 */

template <size_t N>
void runExperiment(std::unique_ptr<Landscape<N>> landscape, const string& name, const string& landscapeCsv) {
  SingleObjectiveGA<N> ga(
      std::move(landscape),
      200,  // popsize
      50,   // generations
      0.05,  // crossover
      0.05,  // mutation
      2,     // elites
      3,     // kParents
      1,      // crossover points
      name
  );
  ga.init();
  auto res = ga.run();
  std::cout << name << " - Accuracy: " << res.getAccuracy() << " Bitstring: " << res.getGene() << std::endl;

  std::ofstream out("pop_" + name + ".csv");
  out << "id,fitness,niche" << std::endl;
  auto niches = ga.findNiches();
  for (size_t i = 0; i < niches.size(); i++) {
    for (auto& ind : niches[i]) {
      out << ind.getGene().to_ulong() << "," << ind.getAccuracy() << "," << i << std::endl;
    }
  }
  out.close();
}

struct HyperParams {
  double crossoverRate;
  double mutationRate;
  int crossoverPoints;
};

template <size_t N>
void runGridSearch(
    std::unique_ptr<Landscape<N>> landscape,
    const string& name,
    const std::vector<HyperParams>& configs,
    int runsPerConfig,
    std::ofstream& perRun,
    std::ofstream& perConfig) {

  SingleObjectiveGA<N> ga(
      std::move(landscape),
      200, 50, 0.05, 0.05, 2, 3, 1, name);

  std::random_device rd;

  for (size_t ci = 0; ci < configs.size(); ci++) {
    const auto& hp = configs[ci];
    ga.setHyperparameters(hp.crossoverRate, hp.mutationRate, hp.crossoverPoints);

    std::vector<double> bestFitnesses;
    bestFitnesses.reserve(runsPerConfig);

    for (int r = 0; r < runsPerConfig; r++) {
      unsigned int seed = rd();
      SingleObjectiveGA<N>::setSeed(seed);

      ga.reset();
      ga.init();
      auto best = ga.run(false);
      double acc = best.getAccuracy();
      bestFitnesses.push_back(acc);

      perRun << name
        << "," << hp.crossoverRate
        << "," << hp.mutationRate
        << "," << hp.crossoverPoints
        << "," << r
        << "," << seed
        << "," << acc
        << "," << best.getGene().to_ulong()
        << "\n";
    }

    double mean = std::accumulate(bestFitnesses.begin(), bestFitnesses.end(), 0.0) / bestFitnesses.size();
    double sqSum = 0.0;
    for (double x : bestFitnesses) sqSum += (x - mean) * (x - mean);
    double stdev = bestFitnesses.size() > 1 ? std::sqrt(sqSum / (bestFitnesses.size() - 1)) : 0.0;
    double best = *std::max_element(bestFitnesses.begin(), bestFitnesses.end());
    double worst = *std::min_element(bestFitnesses.begin(), bestFitnesses.end());

    perConfig << name
      << "," << hp.crossoverRate
      << "," << hp.mutationRate
      << "," << hp.crossoverPoints
      << "," << runsPerConfig
      << "," << mean
      << "," << stdev
      << "," << best
      << "," << worst
      << "\n";

    std::cout << "[" << name << "] cx=" << hp.crossoverRate
      << " mut=" << hp.mutationRate
      << " pts=" << hp.crossoverPoints
      << "  mean=" << mean
      << " std=" << stdev << std::endl;
  }
}

void runSingleObjectiveGAExperiment(int runsPerConfig = 5) {
  double epsilon = 0.05;

  std::vector<int>    crossoverPointValues = {1, 2};
  std::vector<double> crossoverRates       = {0.05, 0.3, 0.7};
  std::vector<double> mutationRates        = {0.01, 0.05, 0.15};

  std::vector<HyperParams> configs;
  for (int cp : crossoverPointValues)
    for (double cx : crossoverRates)
      for (double mut : mutationRates)
        configs.push_back({cx, mut, cp});

  std::cout << "Grid: " << configs.size() << " configs x "
    << runsPerConfig << " runs = "
    << configs.size() * runsPerConfig << " runs per landscape\n";

  std::ofstream perRun("experiment_runs.csv");
  perRun << "landscape,crossover_rate,mutation_rate,crossover_points,"
    << "run,seed,best_accuracy,best_gene\n";

  std::ofstream perConfig("experiment_configs.csv");
  perConfig << "landscape,crossover_rate,mutation_rate,crossover_points,"
    << "runs,mean_accuracy,std_accuracy,best_accuracy,worst_accuracy\n";

  {
    auto landscape = std::make_unique<TriangleLandscape<16>>(1, 4, epsilon);
    landscape->precompute("data/triangle.csv");
    runGridSearch<16>(std::move(landscape), "triangle", configs, runsPerConfig, perRun, perConfig);
  }
  {
    auto landscape = std::make_unique<HDF5Landscape<9>>("data/01-breast-w_lr_F.h5", epsilon);
    runGridSearch<9>(std::move(landscape), "breast", configs, runsPerConfig, perRun, perConfig);
  }
  {
    auto landscape = std::make_unique<HDF5Landscape<15>>("data/05-credit-a_rf_F.h5", epsilon);
    runGridSearch<15>(std::move(landscape), "credit", configs, runsPerConfig, perRun, perConfig);
  }
  {
    auto landscape = std::make_unique<HDF5Landscape<16>>("data/08-letter-r_knn_F.h5", epsilon);
    runGridSearch<16>(std::move(landscape), "letter", configs, runsPerConfig, perRun, perConfig);
  }
=======
>>>>>>> dev

  perRun.close();
  perConfig.close();
  std::cout << "\nWrote experiment_runs.csv and experiment_configs.csv\n";
}

int main() {
  runSingleObjectiveGAExperiment(5);
  return 0;
}

