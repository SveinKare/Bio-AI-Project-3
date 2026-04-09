#include <H5Cpp.h>

#include <bitset>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

template <std::size_t N>
class Landscape {
 public:
  virtual double fitness(std::bitset<N> gene) const = 0;
  virtual ~Landscape() = default;
};

template <std::size_t N>
class HDF5Landscape : public Landscape<N> {
 public:
  explicit HDF5Landscape(
      const std::string& hdf5_path,
      bool is_feature_selection_landscape = true)
      : is_feature_selection_landscape_(is_feature_selection_landscape) {
    static_assert(
        N < (sizeof(std::size_t) * 8),
        "N is too large to convert bitset to size_t index safely.");

    loadFromFile(hdf5_path);
  }

  double fitness(std::bitset<N> gene) const override {
    return fitnessByIndex(geneToIndex(gene));
  }

  double meanTime(std::bitset<N> gene) const {
    return timeByIndex(geneToIndex(gene));
  }

  double fitnessByIndex(std::size_t index) const {
    if (index >= mean_accuracy_.size()) {
      throw std::out_of_range("Fitness index out of range.");
    }
    return mean_accuracy_[index];
  }

  double timeByIndex(std::size_t index) const {
    if (index >= mean_time_.size()) {
      throw std::out_of_range("Time index out of range.");
    }
    return mean_time_[index];
  }

  const std::vector<double>& meanAccuracies() const { return mean_accuracy_; }
  const std::vector<double>& meanTimes() const { return mean_time_; }

  void exportToCSV(const std::string& csv_path) const {
    std::ofstream out(csv_path);
    if (!out.is_open()) {
      throw std::runtime_error("Cannot open CSV file for writing: " + csv_path);
    }

    out << "index,bitmask,num_features,mean_accuracy,mean_time\n";
    out << std::setprecision(10);

    const std::size_t start = is_feature_selection_landscape_ ? 1 : 0;
    for (std::size_t i = start; i < mean_accuracy_.size(); ++i) {
      std::bitset<N> bits(i);
      out << i << ','
          << bits.to_string() << ','
          << bits.count() << ','
          << mean_accuracy_[i] << ','
          << mean_time_[i] << '\n';
    }
  }

 private:
  bool is_feature_selection_landscape_;
  std::vector<double> mean_accuracy_;
  std::vector<double> mean_time_;

  static std::size_t geneToIndex(const std::bitset<N>& gene) {
    std::size_t idx = 0;
    for (std::size_t b = 0; b < N; ++b) {
      if (gene.test(b)) {
        idx |= (std::size_t{1} << b);
      }
    }
    return idx;
  }

  static std::pair<std::vector<float>, std::pair<std::size_t, std::size_t>>
  read2DFloatDataset(const H5::H5File& file, const std::string& dataset_name) {
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();

    constexpr int expected_rank = 2;
    const int rank = dataspace.getSimpleExtentNdims();
    if (rank != expected_rank) {
      throw std::runtime_error("Dataset '" + dataset_name +
                               "' must be 2D, but got rank " +
                               std::to_string(rank) + ".");
    }

    hsize_t dims[expected_rank];
    dataspace.getSimpleExtentDims(dims, nullptr);
    const std::size_t rows = static_cast<std::size_t>(dims[0]);
    const std::size_t cols = static_cast<std::size_t>(dims[1]);

    std::vector<float> values(rows * cols);
    dataset.read(values.data(), H5::PredType::NATIVE_FLOAT);
    return {std::move(values), {rows, cols}};
  }

  void loadFromFile(const std::string& hdf5_path) {
    const std::size_t expected_combinations =
        is_feature_selection_landscape_ ? ((std::size_t{1} << N) - 1)
                                        : (std::size_t{1} << N);
    const std::size_t output_size = std::size_t{1} << N;
    const std::size_t index_offset = is_feature_selection_landscape_ ? 1 : 0;

    H5::H5File file(hdf5_path, H5F_ACC_RDONLY);

    const auto [acc_values, acc_dims] = read2DFloatDataset(file, "accuracies");
    const auto [time_values, time_dims] = read2DFloatDataset(file, "times");

    if (acc_dims != time_dims) {
      throw std::runtime_error(
          "Datasets 'accuracies' and 'times' do not have matching dimensions.");
    }

    const std::size_t rows = acc_dims.first;
    const std::size_t cols = acc_dims.second;

    std::size_t combo_axis = 0;
    std::size_t combinations = 0;
    std::size_t instances = 0;

    if (rows == expected_combinations && cols != expected_combinations) {
      combo_axis = 0;
      combinations = rows;
      instances = cols;
    } else if (cols == expected_combinations && rows != expected_combinations) {
      combo_axis = 1;
      combinations = cols;
      instances = rows;
    } else if (rows == expected_combinations && cols == expected_combinations) {
      combo_axis = 0;
      combinations = rows;
      instances = cols;
    } else {
      throw std::runtime_error(
          "Could not infer combination axis from dataset shape (" +
          std::to_string(rows) + ", " + std::to_string(cols) +
          "). Expected one dimension to be " +
          std::to_string(expected_combinations) + ".");
    }

    mean_accuracy_.assign(output_size, 0.0);
    mean_time_.assign(output_size, 0.0);

    for (std::size_t combo = 0; combo < combinations; ++combo) {
      double acc_sum = 0.0;
      double time_sum = 0.0;

      for (std::size_t instance = 0; instance < instances; ++instance) {
        std::size_t row = 0;
        std::size_t col = 0;
        if (combo_axis == 0) {
          row = combo;
          col = instance;
        } else {
          row = instance;
          col = combo;
        }

        const std::size_t flat_index = row * cols + col;
        acc_sum += static_cast<double>(acc_values[flat_index]);
        time_sum += static_cast<double>(time_values[flat_index]);
      }

      const std::size_t table_index = combo + index_offset;
      mean_accuracy_[table_index] = acc_sum / static_cast<double>(instances);
      mean_time_[table_index] = time_sum / static_cast<double>(instances);
    }
  }
};

template <std::size_t N>
class TriangleLandscape : public Landscape<N> {
 private:
  int n_;
  int m_;
  int s_;

 public:
  TriangleLandscape(int n, int m, int s) : n_(n), m_(m), s_(s) {}

  double fitness(std::bitset<N> gene) const override {
    // Placeholder for the synthetic triangle landscape objective.
    // This keeps the class usable while you implement the final formula.
    (void)gene;
    (void)n_;
    (void)m_;
    (void)s_;
    return 0.0;
  }
};

#include <bitset>

using namespace std;

template<size_t N>
class Landscape {
  public:
    virtual pair<double, double> fitness(bitset<N> gene) const = 0;
    virtual ~Landscape() = default;
};

template<size_t N>
class TriangleLandscape : public Landscape<N> {
  private:
    int m;
    int s;
  public:
    TriangleLandscape(int m, int s): m(m), s(s) {};

    pair<double, double> fitness(bitset<N> gene) const override {
      int b = gene.count();
      int ceil = (b + s - 1) / s; // Cpp trick to get ceiling of integer division

      double penalty = 0.0; // Change later if needed

      // Triangle func
      if (ceil % 2 == 1) {
        // g(b)
        if (b % s == 0) {
          return { m*s, penalty };
        } else {
          return { m*(b % s), penalty };
        }
      } else {
        return { m*(ceil*s - b), penalty };
      }
    }
};
