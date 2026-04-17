#include <bitset>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "landscapes.cpp"
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
    if (!isPowerOfTwo(target)) {
      continue;
    }

    std::size_t n = 0;
    while ((std::size_t{1} << n) < target) {
      ++n;
    }
    if ((std::size_t{1} << n) == target) {
      return n;
    }
  }

  throw std::runtime_error(
      "Could not infer feature count from 'accuracies' shape.");
}

template <std::size_t N>
void runLandscape(const std::string& hdf5_path, const std::string& csv_path) {
  HDF5Landscape<N> landscape(hdf5_path, true);

  const std::bitset<N> only_feature_0(1UL);
  const std::bitset<N> all_features((std::size_t{1} << N) - 1);

  auto [acc1, time1, penalty] = landscape.fitness(only_feature_0);
  auto [acc_all, time_all, penalty_all] = landscape.fitness(all_features);

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

struct Args {
  std::string hdf5_path = "data/01-breast-w_lr_F.h5";
  std::string csv_path;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--export" && i + 1 < argc) {
      args.csv_path = argv[++i];
    } else if (arg[0] != '-') {
      args.hdf5_path = arg;
    }
  }
  return args;
}

}  // namespace
/*
int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
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
            "Unsupported feature count " + std::to_string(feature_count) +
            ". Add a switch case for this N.");
    }
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
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

  perRun.close();
  perConfig.close();
  std::cout << "\nWrote experiment_runs.csv and experiment_configs.csv\n";
}

int main() {
  runSingleObjectiveGAExperiment(5);
  return 0;
}

