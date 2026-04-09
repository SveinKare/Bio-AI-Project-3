#include <bitset>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "landscapes.cpp"
#include "nsga2.hpp"

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
void runLandscape(const std::string& hdf5_path, const std::string& csv_path) {
  HDF5Landscape<N> landscape(hdf5_path, true);

  const std::bitset<N> only_feature_0(1UL);
  const std::bitset<N> all_features((std::size_t{1} << N) - 1);

  auto [acc1, time1] = landscape.fitness(only_feature_0);
  auto [acc_all, time_all] = landscape.fitness(all_features);

  std::cout << "Loaded: " << hdf5_path << '\n';
  std::cout << "Detected features: " << N << '\n';
  std::cout << "Mean accuracy for index 1: " << acc1 << '\n';
  std::cout << "Mean time for index 1: " << time1 << '\n';
  std::cout << "Mean accuracy for all features: " << acc_all << '\n';

  if (!csv_path.empty()) {
    landscape.exportToCSV(csv_path);
    std::cout << "Exported CSV to: " << csv_path << '\n';
  }
}

template <std::size_t N>
void runNSGA2_HDF5(const std::string& hdf5_path,
                   const std::string& out_prefix,
                   int pop_size, int generations, unsigned seed) {
  HDF5Landscape<N> landscape(hdf5_path, true);

  typename NSGA2<N>::Config cfg;
  cfg.pop_size = pop_size;
  cfg.generations = generations;
  cfg.seed = seed;
  cfg.maximize_first = true;
  cfg.minimize_second = true;

  typename NSGA2<N>::EvalFn eval =
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
    return landscape.fitness(chr);
  };

  NSGA2<N> nsga(eval, cfg);

  std::cout << "Running NSGA-II on " << hdf5_path
            << " (N=" << N << ", pop=" << pop_size
            << ", gens=" << generations << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';

  if (!front.empty()) {
    double best_acc = 0, min_time = 1e18;
    for (const auto& ind : front) {
      best_acc = std::max(best_acc, ind.obj_accuracy);
      min_time = std::min(min_time, ind.obj_time);
    }
    std::cout << "  Best accuracy on front: " << best_acc << '\n';
    std::cout << "  Min time on front: " << min_time << '\n';
  }

  nsga.exportParetoCSV(out_prefix + "_pareto.csv");
  nsga.exportGenerationsCSV(out_prefix + "_gens.csv");
  nsga.exportPopulationCSV(out_prefix + "_pop.csv");

  std::cout << "  Exported: " << out_prefix << "_pareto.csv\n";
  std::cout << "  Exported: " << out_prefix << "_gens.csv\n";
  std::cout << "  Exported: " << out_prefix << "_pop.csv\n";
}

template <std::size_t N>
void runNSGA2_Triangle(const std::string& out_prefix,
                       int m, int s,
                       int pop_size, int generations, unsigned seed) {
  TriangleLandscape<N> landscape(m, s);

  typename NSGA2<N>::Config cfg;
  cfg.pop_size = pop_size;
  cfg.generations = generations;
  cfg.seed = seed;
  cfg.maximize_first = true;   // maximize fitness
  cfg.minimize_second = true;  // minimize popcount (num features)

  typename NSGA2<N>::EvalFn eval =
      [&landscape](const std::bitset<N>& chr) -> std::pair<double, double> {
    auto [fit, _penalty] = landscape.fitness(chr);
    double popcount = static_cast<double>(chr.count());
    return {fit, popcount};
  };

  NSGA2<N> nsga(eval, cfg);

  std::cout << "Running NSGA-II on Triangle landscape"
            << " (N=" << N << ", m=" << m << ", s=" << s
            << ", pop=" << pop_size
            << ", gens=" << generations << ", seed=" << seed << ")\n";

  nsga.run();

  auto front = nsga.paretoFront();
  std::cout << "  Pareto front size: " << front.size() << '\n';

  nsga.exportParetoCSV(out_prefix + "_pareto.csv");
  nsga.exportGenerationsCSV(out_prefix + "_gens.csv");
  nsga.exportPopulationCSV(out_prefix + "_pop.csv");

  std::cout << "  Exported: " << out_prefix << "_pareto.csv\n";
  std::cout << "  Exported: " << out_prefix << "_gens.csv\n";
  std::cout << "  Exported: " << out_prefix << "_pop.csv\n";
}

struct Args {
  std::string hdf5_path = "data/01-breast-w_lr_F.h5";
  std::string csv_path;
  std::string out_prefix;
  bool run_nsga2 = false;
  int pop_size = 100;
  int generations = 200;
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
    }else if (arg == "--out" && i + 1 < argc) {
      args.out_prefix = argv[++i];
    } else if (arg == "--pop" && i + 1 < argc) {
      args.pop_size = std::stoi(argv[++i]);
    } else if (arg == "--gens" && i + 1 < argc) {
      args.generations = std::stoi(argv[++i]);
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
      runNSGA2_Triangle<16>(prefix, 1, 4,
                            args.pop_size, args.generations, args.seed);
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
          runNSGA2_HDF5<9>(args.hdf5_path, prefix,
                           args.pop_size, args.generations, args.seed);
          break;
        case 15:
          runNSGA2_HDF5<15>(args.hdf5_path, prefix,
                            args.pop_size, args.generations, args.seed);
          break;
        case 16:
          runNSGA2_HDF5<16>(args.hdf5_path, prefix,
                            args.pop_size, args.generations, args.seed);
          break;
        default:
          throw std::runtime_error(
              "Unsupported feature count " + std::to_string(feature_count));
      }
      return 0;
    }

    // Default: simple landscape lookup
    const std::size_t feature_count =
        inferFeatureCountFromFile(args.hdf5_path, true);

    switch (feature_count) {
      case 9:
        runLandscape<9>(args.hdf5_path, args.csv_path);
        break;
      case 15:
        runLandscape<15>(args.hdf5_path, args.csv_path);
        break;
      case 16:
        runLandscape<16>(args.hdf5_path, args.csv_path);
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
