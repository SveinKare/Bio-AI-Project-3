#include <bitset>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "landscapes.cpp"
#include "nsga2.cpp"

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

template <std::size_t N>
void printBestSolution(const std::vector<Individual<N>>& front) {
  if (front.empty()) return;

  const auto& best = *std::max_element(front.begin(), front.end(),
      [](const auto& a, const auto& b) {
        return a.obj_accuracy < b.obj_accuracy;
      });

  std::cout << "\n  -- Best solution found --\n";
  std::cout << "  Bitstring: " << best.chromosome.to_string() << '\n';
  std::cout << "  Features:  " << best.chromosome.count() << " / " << N << '\n';
  std::cout << "  Accuracy:  " << best.obj_accuracy << '\n';
  std::cout << "  Time:      " << best.obj_time << '\n';
}

template <std::size_t N>
void runNSGA2_HDF5(const std::string& hdf5_path,
                   const std::string& out_prefix,
                   double epsilon, unsigned seed) {
  HDF5Landscape<N> landscape(hdf5_path, epsilon, true);

  typename NSGA2<N>::Config cfg;
  cfg.seed = seed;
  cfg.maximize_first = true;
  cfg.minimize_second = true;

  // NSGA-II uses raw accuracy (no penalty) for true multi-objective optimisation
  typename NSGA2<N>::EvalFn eval =
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
    auto f = landscape.fitness(chr);
    return {f.accuracy, f.mean_time};
  };

  NSGA2<N> nsga(eval, cfg);

  std::cout << "Running NSGA-II on " << hdf5_path
            << " (N=" << N
            << ", pop=" << cfg.pop_size
            << ", gens=" << cfg.generations
            << ", mutation=" << cfg.mutation_rate
            << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';
  printBestSolution<N>(front);

  nsga.exportParetoCSV(out_prefix + "_pareto.csv");
  nsga.exportGenerationsCSV(out_prefix + "_gens.csv");
  nsga.exportPopulationCSV(out_prefix + "_pop.csv");
  nsga.exportSnapshotsCSV(out_prefix + "_snapshots.csv");

  std::cout << "\n  Exported: " << out_prefix << "_pareto.csv\n";
  std::cout << "  Exported: " << out_prefix << "_gens.csv\n";
  std::cout << "  Exported: " << out_prefix << "_pop.csv\n";
  std::cout << "  Exported: " << out_prefix << "_snapshots.csv\n";
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

  // NSGA-II uses raw accuracy (no penalty) for true multi-objective optimisation
  typename NSGA2<N>::EvalFn eval =
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
    auto f = landscape.fitness(chr);
    double popcount = static_cast<double>(chr.count());
    return {f.accuracy, popcount};
  };

  NSGA2<N> nsga(eval, cfg);

  std::cout << "Running NSGA-II on Triangle landscape"
            << " (N=" << N << ", m=" << m << ", s=" << s
            << ", pop=" << cfg.pop_size
            << ", gens=" << cfg.generations
            << ", mutation=" << cfg.mutation_rate
            << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';
  printBestSolution<N>(front);

  nsga.exportParetoCSV(out_prefix + "_pareto.csv");
  nsga.exportGenerationsCSV(out_prefix + "_gens.csv");
  nsga.exportPopulationCSV(out_prefix + "_pop.csv");
  nsga.exportSnapshotsCSV(out_prefix + "_snapshots.csv");

  std::cout << "\n  Exported: " << out_prefix << "_pareto.csv\n";
  std::cout << "  Exported: " << out_prefix << "_gens.csv\n";
  std::cout << "  Exported: " << out_prefix << "_pop.csv\n";
  std::cout << "  Exported: " << out_prefix << "_snapshots.csv\n";
}

struct Args {
  std::string hdf5_path = "data/01-breast-w_lr_F.h5";
  std::string csv_path;
  std::string out_prefix;
  bool run_nsga2 = false;
  bool run_triangle = false;
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
    } else if (arg == "--triangle") {
      args.run_triangle = true;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);

    if (args.run_triangle) {
      std::string prefix = args.out_prefix.empty() ? "output/triangle" : args.out_prefix;
      runNSGA2_Triangle<16>(prefix, 1, 4, args.epsilon, args.seed);
      return 0;
    }

    if (args.run_nsga2) {
      const std::size_t feature_count =
          inferFeatureCountFromFile(args.hdf5_path, true);
      std::string prefix = args.out_prefix;
      if (prefix.empty()) {
        prefix = "output/nsga2";
      }

      switch (feature_count) {
        case 9:
          runNSGA2_HDF5<9>(args.hdf5_path, prefix, args.epsilon, args.seed);
          break;
        case 15:
          runNSGA2_HDF5<15>(args.hdf5_path, prefix, args.epsilon, args.seed);
          break;
        case 16:
          runNSGA2_HDF5<16>(args.hdf5_path, prefix, args.epsilon, args.seed);
          break;
        default:
          throw std::runtime_error(
              "Unsupported feature count " + std::to_string(feature_count));
      }
      return 0;
    }

    const std::size_t feature_count =
        inferFeatureCountFromFile(args.hdf5_path, true);

    switch (feature_count) {
      case 9:
        runLandscape<9>(args.hdf5_path, args.csv_path, args.epsilon);
        break;
      case 15:
        runLandscape<15>(args.hdf5_path, args.csv_path, args.epsilon);
        break;
      case 16:
        runLandscape<16>(args.hdf5_path, args.csv_path, args.epsilon);
        break;
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
