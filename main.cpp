#include <bitset>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "landscapes.cpp"
#include "nsga2.cpp"
#include "pso.cpp"
#include "single_ga.cpp"

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

// ── PSO runners ─────────────────────────────────────────────────────────────

template <std::size_t N>
void runPSO_HDF5(const std::string& hdf5_path,
                 const std::string& out_prefix,
                 double epsilon, unsigned seed) {
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
  exportCSVs(pso, out_prefix);
}

template <std::size_t N>
void runPSO_Triangle(const std::string& out_prefix,
                     int m, int s,
                     double epsilon, unsigned seed) {
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
  exportCSVs(pso, out_prefix);
}

// ── NSGA-II runners ─────────────────────────────────────────────────────────

template <std::size_t N>
void runNSGA2_HDF5(const std::string& hdf5_path,
                   const std::string& out_prefix,
                   double epsilon, unsigned seed) {
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
  exportCSVs(nsga, out_prefix);
}

template <std::size_t N>
void runNSGA2_Triangle(const std::string& out_prefix,
                       int m, int s,
                       double epsilon, unsigned seed) {
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
  exportCSVs(nsga, out_prefix);
}

// ── Single-objective GA (single-ga branch workflow / plotting outputs) ────

template <std::size_t N>
void runExperiment(std::unique_ptr<Landscape<N>> landscape,
                   const std::string& name) {
  SingleObjectiveGA<N> ga(
      std::move(landscape),
      200,   // popsize
      50,    // generations
      0.05,  // crossover
      0.05,  // mutation
      2,     // elites
      3,     // kParents
      1,     // crossover points
      name);
  ga.init();
  auto res = ga.run();
  std::cout << name << " - Accuracy: " << res.getAccuracy() << " Bitstring: "
            << res.getGene() << std::endl;

  std::ofstream out("pop_" + name + ".csv");
  out << "id,fitness,niche" << std::endl;
  auto niches = ga.findNiches();
  for (size_t i = 0; i < niches.size(); i++) {
    for (auto& ind : niches[i]) {
      out << ind.getGene().to_ulong() << "," << ind.getAccuracy() << "," << i
          << std::endl;
    }
  }
}

void runSingleGAExperiments(double epsilon) {
  {
    auto landscape = std::make_unique<TriangleLandscape<16>>(1, 4, epsilon);
    landscape->precompute("data/triangle.csv");
    runExperiment<16>(std::move(landscape), "triangle");
  }

  {
    auto landscape =
        std::make_unique<HDF5Landscape<9>>("data/01-breast-w_lr_F.h5", epsilon);
    landscape->exportToCSV("data/landscape_breast.csv");
    runExperiment<9>(std::move(landscape), "breast");
  }

  {
    auto landscape =
        std::make_unique<HDF5Landscape<15>>("data/05-credit-a_rf_F.h5", epsilon);
    landscape->exportToCSV("data/landscape_credit.csv");
    runExperiment<15>(std::move(landscape), "credit");
  }

  {
    auto landscape =
        std::make_unique<HDF5Landscape<16>>("data/08-letter-r_knn_F.h5", epsilon);
    landscape->exportToCSV("data/landscape_letter.csv");
    runExperiment<16>(std::move(landscape), "letter");
  }
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Args {
  std::string hdf5_path = "data/01-breast-w_lr_F.h5";
  std::string csv_path;
  std::string out_prefix;
  bool run_nsga2 = false;
  bool run_pso = false;
  bool run_triangle = false;
  bool run_sga = false;
  double epsilon = 0.1;
  unsigned seed = 42;
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
  if (args.run_triangle) {
    if (args.run_pso)
      runPSO_Triangle<N>(prefix.empty() ? "output/triangle" : prefix,
                         1, 4, args.epsilon, args.seed);
    else
      runNSGA2_Triangle<N>(prefix.empty() ? "output/triangle" : prefix,
                           1, 4, args.epsilon, args.seed);
  } else if (args.run_pso) {
    runPSO_HDF5<N>(args.hdf5_path,
                   prefix.empty() ? "output/pso" : prefix,
                   args.epsilon, args.seed);
  } else if (args.run_nsga2) {
    runNSGA2_HDF5<N>(args.hdf5_path,
                     prefix.empty() ? "output/nsga2" : prefix,
                     args.epsilon, args.seed);
  } else {
    runLandscape<N>(args.hdf5_path, args.csv_path, args.epsilon);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);

    if (args.run_sga) {
      runSingleGAExperiments(args.epsilon);
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

  return 0;
}
