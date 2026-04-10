#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

struct PSOEvalResult {
  double fitness;
  double accuracy;
  double time;
};

template <std::size_t N>
struct Particle {
  std::bitset<N> position;
  std::array<double, N> velocity{};

  double fitness = -std::numeric_limits<double>::infinity();
  double obj_accuracy = 0.0;
  double obj_time = 0.0;

  std::bitset<N> pbest_position;
  double pbest_fitness = -std::numeric_limits<double>::infinity();

  // For CSV output compatibility with NSGA-II plotting scripts.
  int rank = 0;
  double crowding_distance = 0.0;
};

struct PSOIterStats {
  int iteration;
  double gbest_fitness;
  double best_accuracy;
  double min_time;
  double mean_accuracy;
  double mean_time;
  double diversity;
  double mutation_rate;
};

template <std::size_t N>
class BinaryPSO {
 public:
  struct Config {
    int swarm_size = static_cast<int>(2 * N * N);
    int iterations = 100;
    double w_start = 0.9;
    double w_end = 0.4;
    double c1 = 2.0;
    double c2 = 2.0;
    double v_max = 6.0;
    unsigned seed = 42;

    // Mutation decays linearly from mut_start to mut_end (parameter control).
    // If diversity drops below diversity_low, mutation is boosted to prevent
    // premature convergence (adaptive parameter control).
    double mut_start = 1.0 / static_cast<double>(N);
    double mut_end = 0.0;
    double diversity_low = 0.25;
    double mutation_boost = 1.5;
    double mutation_max = 0.5;
  };

  using EvalFn = std::function<PSOEvalResult(const std::bitset<N>&)>;

  BinaryPSO(EvalFn eval, Config cfg)
      : eval_(std::move(eval)), cfg_(cfg), rng_(cfg.seed) {}

  void run() {
    swarm_ = initializeSwarm();
    evaluate(swarm_);
    initializePersonalBests();
    findGlobalBest();

    double current_mut = cfg_.mut_start;

    for (int iter = 0; iter < cfg_.iterations; ++iter) {
      double t = (cfg_.iterations > 1)
          ? static_cast<double>(iter) /
            static_cast<double>(cfg_.iterations - 1)
          : 0.0;

      double w = cfg_.w_start + t * (cfg_.w_end - cfg_.w_start);
      double scheduled_mut = cfg_.mut_start + t * (cfg_.mut_end - cfg_.mut_start);

      for (auto& p : swarm_) {
        updateVelocity(p, w);
        updatePosition(p);
        mutate(p, current_mut);
      }

      evaluate(swarm_);
      updatePersonalBests();
      findGlobalBest();

      double div = measureDiversity();

      if (div < cfg_.diversity_low) {
        current_mut = std::min(current_mut * cfg_.mutation_boost,
                               cfg_.mutation_max);
      } else {
        current_mut = std::max(scheduled_mut, current_mut * 0.95);
      }

      assignRanksForOutput();
      iter_stats_.push_back(computeStats(iter, div, current_mut));
      snapshots_.push_back(swarm_);
    }
  }

  const std::vector<Particle<N>>& swarm() const { return swarm_; }
  const std::vector<PSOIterStats>& iterationStats() const {
    return iter_stats_;
  }
  const Particle<N>& globalBest() const { return gbest_; }

  // ── CSV exports ──────────────────────────────────────────────────────────
  // Column names match the NSGA-II output format so the same plotting scripts
  // can be reused.  "hypervolume" carries the scalar gbest fitness instead.

  void exportParetoCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "index,bitmask,num_features,accuracy,time,rank,crowding_distance\n";
    out << std::setprecision(10);

    out << bitsetToIndex(gbest_.position) << ','
        << gbest_.position.to_string() << ','
        << gbest_.position.count() << ','
        << gbest_.obj_accuracy << ','
        << gbest_.obj_time << ','
        << 0 << ',' << 0 << '\n';
  }

  void exportGenerationsCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "generation,num_fronts,pareto_size,best_accuracy,min_time,"
           "mean_accuracy,mean_time,hypervolume,mutation_rate,diversity\n";
    out << std::setprecision(10);

    for (const auto& s : iter_stats_) {
      out << s.iteration << ','
          << 1 << ',' << 1 << ','
          << s.best_accuracy << ','
          << s.min_time << ','
          << s.mean_accuracy << ','
          << s.mean_time << ','
          << s.gbest_fitness << ','
          << s.mutation_rate << ','
          << s.diversity << '\n';
    }
  }

  void exportPopulationCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "index,bitmask,num_features,accuracy,time,rank,crowding_distance\n";
    out << std::setprecision(10);

    for (const auto& p : swarm_) {
      out << bitsetToIndex(p.position) << ','
          << p.position.to_string() << ','
          << p.position.count() << ','
          << p.obj_accuracy << ','
          << p.obj_time << ','
          << p.rank << ','
          << p.crowding_distance << '\n';
    }
  }

  void exportSnapshotsCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "generation,index,bitmask,num_features,accuracy,time,rank,"
           "crowding_distance\n";
    out << std::setprecision(10);

    for (std::size_t gen = 0; gen < snapshots_.size(); ++gen) {
      for (const auto& p : snapshots_[gen]) {
        out << gen << ','
            << bitsetToIndex(p.position) << ','
            << p.position.to_string() << ','
            << p.position.count() << ','
            << p.obj_accuracy << ','
            << p.obj_time << ','
            << p.rank << ','
            << p.crowding_distance << '\n';
      }
    }
  }

 private:
  EvalFn eval_;
  Config cfg_;
  std::mt19937 rng_;
  std::vector<Particle<N>> swarm_;
  Particle<N> gbest_;
  std::vector<PSOIterStats> iter_stats_;
  std::vector<std::vector<Particle<N>>> snapshots_;

  static std::size_t bitsetToIndex(const std::bitset<N>& bs) {
    std::size_t idx = 0;
    for (std::size_t b = 0; b < N; ++b) {
      if (bs.test(b)) idx |= (std::size_t{1} << b);
    }
    return idx;
  }

  std::vector<Particle<N>> initializeSwarm() {
    std::vector<Particle<N>> s(cfg_.swarm_size);
    std::uniform_int_distribution<int> bit_dist(0, 1);
    std::uniform_real_distribution<double> vel_dist(-1.0, 1.0);

    for (auto& p : s) {
      for (std::size_t b = 0; b < N; ++b) {
        if (bit_dist(rng_)) p.position.set(b);
        p.velocity[b] = vel_dist(rng_);
      }
      if (p.position.none()) {
        p.position.set(rng_() % N);
      }
    }
    return s;
  }

  void evaluate(std::vector<Particle<N>>& particles) {
    for (auto& p : particles) {
      auto result = eval_(p.position);
      p.fitness = result.fitness;
      p.obj_accuracy = result.accuracy;
      p.obj_time = result.time;
    }
  }

  void initializePersonalBests() {
    for (auto& p : swarm_) {
      p.pbest_position = p.position;
      p.pbest_fitness = p.fitness;
    }
  }

  void updatePersonalBests() {
    for (auto& p : swarm_) {
      if (p.fitness > p.pbest_fitness) {
        p.pbest_position = p.position;
        p.pbest_fitness = p.fitness;
      }
    }
  }

  void findGlobalBest() {
    for (const auto& p : swarm_) {
      if (p.fitness > gbest_.fitness) {
        gbest_ = p;
      }
    }
  }

  void updateVelocity(Particle<N>& p, double w) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
      double r1 = dist(rng_);
      double r2 = dist(rng_);

      double pbest_diff = (p.pbest_position.test(i) ? 1.0 : 0.0) -
                           (p.position.test(i) ? 1.0 : 0.0);
      double gbest_diff = (gbest_.position.test(i) ? 1.0 : 0.0) -
                           (p.position.test(i) ? 1.0 : 0.0);

      p.velocity[i] = w * p.velocity[i] +
                       cfg_.c1 * r1 * pbest_diff +
                       cfg_.c2 * r2 * gbest_diff;

      p.velocity[i] =
          std::max(-cfg_.v_max, std::min(cfg_.v_max, p.velocity[i]));
    }
  }

  // V-shaped transfer: P(flip) = |tanh(v)|.
  // Near-zero velocity preserves bits; high velocity flips them.
  void updatePosition(Particle<N>& p) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) {
      if (dist(rng_) < std::abs(std::tanh(p.velocity[i]))) {
        p.position.flip(i);
      }
    }
    if (p.position.none()) {
      p.position.set(rng_() % N);
    }
  }

  void mutate(Particle<N>& p, double rate) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    for (std::size_t b = 0; b < N; ++b) {
      if (prob(rng_) < rate) {
        p.position.flip(b);
      }
    }
    if (p.position.none()) {
      p.position.set(rng_() % N);
    }
  }

  void assignRanksForOutput() {
    std::vector<int> indices(swarm_.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return swarm_[a].fitness > swarm_[b].fitness;
    });

    double best_fit = swarm_[indices[0]].fitness;
    for (int r = 0; r < static_cast<int>(indices.size()); ++r) {
      swarm_[indices[r]].rank = (swarm_[indices[r]].fitness == best_fit) ? 0 : r;
      swarm_[indices[r]].crowding_distance = 0.0;
    }
  }

  double measureDiversity() const {
    std::set<std::string> unique;
    for (const auto& p : swarm_) {
      unique.insert(p.position.to_string());
    }
    return static_cast<double>(unique.size()) /
           static_cast<double>(swarm_.size());
  }

  PSOIterStats computeStats(int iter, double diversity,
                            double mutation_rate) const {
    PSOIterStats s{};
    s.iteration = iter;
    s.gbest_fitness = gbest_.fitness;
    s.diversity = diversity;
    s.mutation_rate = mutation_rate;

    double sum_acc = 0, sum_time = 0;
    s.best_accuracy = -std::numeric_limits<double>::infinity();
    s.min_time = std::numeric_limits<double>::infinity();

    for (const auto& p : swarm_) {
      sum_acc += p.obj_accuracy;
      sum_time += p.obj_time;
      if (p.obj_accuracy > s.best_accuracy) s.best_accuracy = p.obj_accuracy;
      if (p.obj_time < s.min_time) s.min_time = p.obj_time;
    }

    s.mean_accuracy = sum_acc / static_cast<double>(swarm_.size());
    s.mean_time = sum_time / static_cast<double>(swarm_.size());
    return s;
  }
};
