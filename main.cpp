#include <bitset>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "landscapes.hpp"

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

  std::cout << "Loaded: " << hdf5_path << '\n';
  std::cout << "Detected features: " << N << '\n';
  std::cout << "Mean accuracy for index 1: "
            << landscape.fitness(only_feature_0) << '\n';
  std::cout << "Mean time for index 1: "
            << landscape.meanTime(only_feature_0) << '\n';
  std::cout << "Mean accuracy for all features: "
            << landscape.fitness(all_features) << '\n';

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

  return 0;
}
