#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

template <std::size_t N>
struct Individual {
  std::bitset<N> chromosome;
  double obj_accuracy = 0.0;
  double obj_time = 0.0;
  int rank = 0;
  double crowding_distance = 0.0;
};

struct GenerationStats {
  int generation;
  int num_fronts;
  double best_accuracy;
  double min_time;
  double mean_accuracy;
  double mean_time;
  double hypervolume;
  int pareto_size;
};

template <std::size_t N>
class NSGA2 {
 public:
  struct Config {
    int pop_size = 100;
    int generations = 200;
    double crossover_rate = 0.9;
    double mutation_rate = 1.0 / static_cast<double>(N);
    unsigned seed = 42;
    bool maximize_first = true;
    bool minimize_second = true;
  };

  using EvalFn = std::function<std::pair<double, double>(const std::bitset<N>&)>;

  NSGA2(EvalFn eval, Config cfg)
      : eval_(std::move(eval)), cfg_(cfg), rng_(cfg.seed) {}

  void run() {
    population_ = initializePopulation();
    evaluate(population_);

    for (int gen = 0; gen < cfg_.generations; ++gen) {
      auto offspring = createOffspring(population_);
      evaluate(offspring);

      std::vector<Individual<N>> combined;
      combined.reserve(population_.size() + offspring.size());
      combined.insert(combined.end(), population_.begin(), population_.end());
      combined.insert(combined.end(), offspring.begin(), offspring.end());

      auto fronts = nonDominatedSort(combined);
      assignCrowdingDistance(combined, fronts);

      population_ = selectNextGeneration(combined, fronts);

      gen_stats_.push_back(computeStats(gen, population_, fronts));
    }

    auto fronts = nonDominatedSort(population_);
    assignCrowdingDistance(population_, fronts);
  }

  const std::vector<Individual<N>>& population() const { return population_; }
  const std::vector<GenerationStats>& generationStats() const { return gen_stats_; }

  std::vector<Individual<N>> paretoFront() const {
    std::vector<Individual<N>> front;
    for (const auto& ind : population_) {
      if (ind.rank == 0) front.push_back(ind);
    }
    return front;
  }

  void exportParetoCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "index,bitmask,num_features,accuracy,time,rank,crowding_distance\n";
    out << std::setprecision(10);

    auto front = paretoFront();
    std::sort(front.begin(), front.end(),
              [](const auto& a, const auto& b) {
                return a.obj_accuracy > b.obj_accuracy;
              });

    for (const auto& ind : front) {
      std::size_t idx = bitsetToIndex(ind.chromosome);
      out << idx << ','
          << ind.chromosome.to_string() << ','
          << ind.chromosome.count() << ','
          << ind.obj_accuracy << ','
          << ind.obj_time << ','
          << ind.rank << ','
          << ind.crowding_distance << '\n';
    }
  }

  void exportGenerationsCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "generation,num_fronts,pareto_size,best_accuracy,min_time,"
           "mean_accuracy,mean_time,hypervolume\n";
    out << std::setprecision(10);

    for (const auto& s : gen_stats_) {
      out << s.generation << ','
          << s.num_fronts << ','
          << s.pareto_size << ','
          << s.best_accuracy << ','
          << s.min_time << ','
          << s.mean_accuracy << ','
          << s.mean_time << ','
          << s.hypervolume << '\n';
    }
  }

  void exportPopulationCSV(const std::string& path) const {
    std::ofstream out(path);
    out << "index,bitmask,num_features,accuracy,time,rank,crowding_distance\n";
    out << std::setprecision(10);

    for (const auto& ind : population_) {
      std::size_t idx = bitsetToIndex(ind.chromosome);
      out << idx << ','
          << ind.chromosome.to_string() << ','
          << ind.chromosome.count() << ','
          << ind.obj_accuracy << ','
          << ind.obj_time << ','
          << ind.rank << ','
          << ind.crowding_distance << '\n';
    }
  }

 private:
  EvalFn eval_;
  Config cfg_;
  std::mt19937 rng_;
  std::vector<Individual<N>> population_;
  std::vector<GenerationStats> gen_stats_;

  static std::size_t bitsetToIndex(const std::bitset<N>& bs) {
    std::size_t idx = 0;
    for (std::size_t b = 0; b < N; ++b) {
      if (bs.test(b)) idx |= (std::size_t{1} << b);
    }
    return idx;
  }

  std::vector<Individual<N>> initializePopulation() {
    std::vector<Individual<N>> pop(cfg_.pop_size);
    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (auto& ind : pop) {
      for (std::size_t b = 0; b < N; ++b) {
        if (bit_dist(rng_)) ind.chromosome.set(b);
      }
      if (ind.chromosome.none()) {
        ind.chromosome.set(rng_() % N);
      }
    }
    return pop;
  }

  void evaluate(std::vector<Individual<N>>& pop) {
    for (auto& ind : pop) {
      auto [acc, time] = eval_(ind.chromosome);
      ind.obj_accuracy = acc;
      ind.obj_time = time;
    }
  }

  bool dominates(const Individual<N>& a, const Individual<N>& b) const {
    bool at_least_one_better = false;

    if (cfg_.maximize_first) {
      if (a.obj_accuracy < b.obj_accuracy) return false;
      if (a.obj_accuracy > b.obj_accuracy) at_least_one_better = true;
    } else {
      if (a.obj_accuracy > b.obj_accuracy) return false;
      if (a.obj_accuracy < b.obj_accuracy) at_least_one_better = true;
    }

    if (cfg_.minimize_second) {
      if (a.obj_time > b.obj_time) return false;
      if (a.obj_time < b.obj_time) at_least_one_better = true;
    } else {
      if (a.obj_time < b.obj_time) return false;
      if (a.obj_time > b.obj_time) at_least_one_better = true;
    }

    return at_least_one_better;
  }

  std::vector<std::vector<int>> nonDominatedSort(
      std::vector<Individual<N>>& pop) const {
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> dominated_by(n);
    std::vector<int> domination_count(n, 0);
    std::vector<std::vector<int>> fronts;

    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (dominates(pop[i], pop[j])) {
          dominated_by[i].push_back(j);
          domination_count[j]++;
        } else if (dominates(pop[j], pop[i])) {
          dominated_by[j].push_back(i);
          domination_count[i]++;
        }
      }
    }

    std::vector<int> front0;
    for (int i = 0; i < n; ++i) {
      if (domination_count[i] == 0) {
        pop[i].rank = 0;
        front0.push_back(i);
      }
    }
    fronts.push_back(std::move(front0));

    int current_rank = 0;
    while (!fronts[current_rank].empty()) {
      std::vector<int> next_front;
      for (int i : fronts[current_rank]) {
        for (int j : dominated_by[i]) {
          domination_count[j]--;
          if (domination_count[j] == 0) {
            pop[j].rank = current_rank + 1;
            next_front.push_back(j);
          }
        }
      }
      current_rank++;
      fronts.push_back(std::move(next_front));
    }

    fronts.pop_back();
    return fronts;
  }

  void assignCrowdingDistance(std::vector<Individual<N>>& pop,
                              const std::vector<std::vector<int>>& fronts) const {
    for (auto& ind : pop) ind.crowding_distance = 0.0;

    for (const auto& front : fronts) {
      const int fsize = static_cast<int>(front.size());
      if (fsize <= 2) {
        for (int idx : front) {
          pop[idx].crowding_distance = std::numeric_limits<double>::infinity();
        }
        continue;
      }

      auto sortAndAccum = [&](auto objGetter) {
        std::vector<int> sorted_idx(front.begin(), front.end());
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) {
                    return objGetter(pop[a]) < objGetter(pop[b]);
                  });

        double obj_min = objGetter(pop[sorted_idx.front()]);
        double obj_max = objGetter(pop[sorted_idx.back()]);
        double range = obj_max - obj_min;

        pop[sorted_idx.front()].crowding_distance =
            std::numeric_limits<double>::infinity();
        pop[sorted_idx.back()].crowding_distance =
            std::numeric_limits<double>::infinity();

        if (range > 0.0) {
          for (int k = 1; k < fsize - 1; ++k) {
            double dist = (objGetter(pop[sorted_idx[k + 1]]) -
                           objGetter(pop[sorted_idx[k - 1]])) /
                          range;
            pop[sorted_idx[k]].crowding_distance += dist;
          }
        }
      };

      sortAndAccum([](const Individual<N>& ind) { return ind.obj_accuracy; });
      sortAndAccum([](const Individual<N>& ind) { return ind.obj_time; });
    }
  }

  Individual<N> tournamentSelect(const std::vector<Individual<N>>& pop) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pop.size()) - 1);
    int a = dist(rng_);
    int b = dist(rng_);

    if (pop[a].rank < pop[b].rank) return pop[a];
    if (pop[b].rank < pop[a].rank) return pop[b];
    return (pop[a].crowding_distance >= pop[b].crowding_distance) ? pop[a] : pop[b];
  }

  std::pair<Individual<N>, Individual<N>> crossover(
      const Individual<N>& p1, const Individual<N>& p2) {
    Individual<N> c1, c2;
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    if (prob(rng_) < cfg_.crossover_rate) {
      for (std::size_t b = 0; b < N; ++b) {
        if (prob(rng_) < 0.5) {
          if (p1.chromosome.test(b)) c1.chromosome.set(b);
          if (p2.chromosome.test(b)) c2.chromosome.set(b);
        } else {
          if (p2.chromosome.test(b)) c1.chromosome.set(b);
          if (p1.chromosome.test(b)) c2.chromosome.set(b);
        }
      }
    } else {
      c1.chromosome = p1.chromosome;
      c2.chromosome = p2.chromosome;
    }

    return {c1, c2};
  }

  void mutate(Individual<N>& ind) {
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    for (std::size_t b = 0; b < N; ++b) {
      if (prob(rng_) < cfg_.mutation_rate) {
        ind.chromosome.flip(b);
      }
    }
    if (ind.chromosome.none()) {
      ind.chromosome.set(rng_() % N);
    }
  }

  std::vector<Individual<N>> createOffspring(
      const std::vector<Individual<N>>& parents) {
    std::vector<Individual<N>> offspring;
    offspring.reserve(cfg_.pop_size);

    while (static_cast<int>(offspring.size()) < cfg_.pop_size) {
      auto p1 = tournamentSelect(parents);
      auto p2 = tournamentSelect(parents);
      auto [c1, c2] = crossover(p1, p2);
      mutate(c1);
      mutate(c2);
      offspring.push_back(std::move(c1));
      if (static_cast<int>(offspring.size()) < cfg_.pop_size) {
        offspring.push_back(std::move(c2));
      }
    }

    return offspring;
  }

  std::vector<Individual<N>> selectNextGeneration(
      std::vector<Individual<N>>& combined,
      const std::vector<std::vector<int>>& fronts) {
    std::vector<Individual<N>> next_gen;
    next_gen.reserve(cfg_.pop_size);

    for (const auto& front : fronts) {
      if (static_cast<int>(next_gen.size()) + static_cast<int>(front.size()) <=
          cfg_.pop_size) {
        for (int idx : front) {
          next_gen.push_back(combined[idx]);
        }
      } else {
        std::vector<int> sorted_front(front.begin(), front.end());
        std::sort(sorted_front.begin(), sorted_front.end(),
                  [&](int a, int b) {
                    return combined[a].crowding_distance >
                           combined[b].crowding_distance;
                  });

        int remaining = cfg_.pop_size - static_cast<int>(next_gen.size());
        for (int k = 0; k < remaining; ++k) {
          next_gen.push_back(combined[sorted_front[k]]);
        }
        break;
      }
    }

    return next_gen;
  }

  double computeHypervolume(const std::vector<Individual<N>>& pop,
                            const std::vector<std::vector<int>>& fronts) const {
    if (fronts.empty() || fronts[0].empty()) return 0.0;

    double ref_time = 0.0;
    for (const auto& ind : pop) {
      ref_time = std::max(ref_time, ind.obj_time);
    }
    ref_time *= 1.1;

    std::vector<std::pair<double, double>> points;
    for (int idx : fronts[0]) {
      points.emplace_back(pop[idx].obj_accuracy, pop[idx].obj_time);
    }

    std::sort(points.begin(), points.end(),
              [](const auto& a, const auto& b) {
                return a.first > b.first;
              });

    double hv = 0.0;
    double prev_time = ref_time;
    for (const auto& [acc, t] : points) {
      if (t < prev_time) {
        hv += acc * (prev_time - t);
        prev_time = t;
      }
    }

    return hv;
  }

  GenerationStats computeStats(
      int gen,
      const std::vector<Individual<N>>& pop,
      const std::vector<std::vector<int>>& fronts) const {
    GenerationStats s{};
    s.generation = gen;
    s.num_fronts = static_cast<int>(fronts.size());

    double sum_acc = 0, sum_time = 0;
    s.best_accuracy = -std::numeric_limits<double>::infinity();
    s.min_time = std::numeric_limits<double>::infinity();

    int pareto_count = 0;
    for (const auto& ind : pop) {
      sum_acc += ind.obj_accuracy;
      sum_time += ind.obj_time;
      if (ind.obj_accuracy > s.best_accuracy) s.best_accuracy = ind.obj_accuracy;
      if (ind.obj_time < s.min_time) s.min_time = ind.obj_time;
      if (ind.rank == 0) pareto_count++;
    }

    s.mean_accuracy = sum_acc / static_cast<double>(pop.size());
    s.mean_time = sum_time / static_cast<double>(pop.size());
    s.pareto_size = pareto_count;
    s.hypervolume = computeHypervolume(pop, fronts);

    return s;
  }
};
