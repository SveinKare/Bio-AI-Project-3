#include <bitset>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "landscapes.cpp"
#include "single_ga.cpp"
#include <memory>

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

int main() {
  double epsilon = 0.05;
  // Triangle landscape (N=16)
  {
    auto landscape = std::make_unique<TriangleLandscape<16>>(1, 4, epsilon);
    landscape->precompute("data/triangle.csv");
    runExperiment<16>(std::move(landscape), "triangle", "data/triangle.csv");
  }

  // Breast cancer (check your feature count - likely N=9)
  {
    auto landscape = std::make_unique<HDF5Landscape<9>>("data/01-breast-w_lr_F.h5", epsilon);
    landscape->exportToCSV("data/landscape_breast.csv");
    runExperiment<9>(std::move(landscape), "breast", "data/landscape_breast.csv");
  }

  // Credit (likely N=15)
  {
    auto landscape = std::make_unique<HDF5Landscape<15>>("data/05-credit-a_rf_F.h5", epsilon);
    landscape->exportToCSV("data/landscape_credit.csv");
    runExperiment<15>(std::move(landscape), "credit", "data/landscape_credit.csv");
  }

  // Letter recognition (likely N=16)
  {
    auto landscape = std::make_unique<HDF5Landscape<16>>("data/08-letter-r_knn_F.h5", epsilon);
    landscape->exportToCSV("data/landscape_letter.csv");
    runExperiment<16>(std::move(landscape), "letter", "data/landscape_letter.csv");
  }

  return 0;
}
