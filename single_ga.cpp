#include "landscapes.cpp"
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <iostream>
#include <map>

using namespace std;

mt19937 gen(1234);
bernoulli_distribution berRand(0.5);

void seedSgaRng(unsigned s) { gen.seed(s); }

template <size_t N>
class SgaIndividual {
  private: 
    Fitness fitness;
    bitset<N> gene;

  public:
    SgaIndividual() = default;

    SgaIndividual(bitset<N> gene): gene(gene) {}

    SgaIndividual(bitset<N> gene, Fitness fitness): gene(gene), fitness(fitness) {}

    double getFitness() const { return fitness.accuracy-fitness.penalty; }

    void setFitness(Fitness fitness) { this->fitness = fitness; }

    double getAccuracy() const { return fitness.accuracy; }

    int getActiveFeatures() const { return gene.count(); }

    bitset<N> getGene() const { return gene; }

    void setGene(bitset<N> gene) { this->gene = gene; }

    void mutate(double mutationRate) {
      std::uniform_real_distribution<double> mutate(0.0, 1.0);
      for (size_t i = 0; i < this->gene.size(); ++i) {
        if (mutate(gen) < mutationRate) {
          this->gene.flip(i);
        }
      }
    }
};

template <size_t N>
int hammingDist(SgaIndividual<N> &a, SgaIndividual<N> &b) {
  return (a.getGene() ^ b.getGene()).count();
}

template<size_t N>
class SingleObjectiveGA {
  private:
    unique_ptr<Landscape<N>> landscape;
    int popSize;
    int generations;
    double crossoverRate;
    double mutationRate;
    int numberOfElites;
    int kParents;
    int crossoverPoints;
    string name;
    /** If non-empty, stats and anim frames are written under this directory. */
    string outputDir_;

    vector<SgaIndividual<N>> population;

    bitset<N> randomGene() {
      bitset<N> gene;
      gene.reset();

      for (size_t i = 0; i < N; i++) {
        gene.set(i, berRand(gen));
      }

      return gene;
    }

  public:
    SingleObjectiveGA(
        unique_ptr<Landscape<N>> landscape,
        int popSize,
        int generations,
        double crossoverRate,
        double mutationRate,
        int numberOfElites,
        int kParents,
        int crossoverPoints,
        const string& name,
        const string& output_dir = {}
        ): 
      landscape(std::move(landscape)), 
      popSize(popSize), 
      generations(generations), 
      crossoverRate(crossoverRate),
      mutationRate(mutationRate),
      numberOfElites(numberOfElites),
      kParents(kParents),
      crossoverPoints(crossoverPoints),
      name(name),
      outputDir_(output_dir)
  {}

    vector<SgaIndividual<N>> getPopulation() const { return this->population; }

    void reset() {
      population.clear();
    }

    static void setSeed(unsigned int seed) {
      gen.seed(seed);
    }

    void setHyperparameters(double crossoverRate, double mutationRate, int crossoverPoints) {
      this->crossoverRate = crossoverRate;
      this->mutationRate = mutationRate;
      this->crossoverPoints = crossoverPoints;
    }

    void dumpPopulation(int generation, const string& name) {
      std::ofstream out("anim/" + name + "_gen_" + std::to_string(generation) + ".csv");
      out << "id,fitness" << std::endl;
      for (auto& ind : this->population) {
        out << ind.getGene().to_ulong() << "," << ind.getAccuracy() << std::endl;
      }
      out.close();
    }

    void dumpGeneration(int generation, const string& name) {
      double totalDist = 0;
      int distCount = 0;
      for (size_t i = 0; i < population.size(); i++) {
        for (size_t j = i + 1; j < population.size(); j++) {
          totalDist += (population[i].getGene() ^ population[j].getGene()).count();
          distCount++;
        }
      }
      double avgDist = distCount > 0 ? totalDist / distCount : 0;

      set<string> unique;
      for (auto& ind : population) {
        unique.insert(ind.getGene().to_string());
      }

      double entropy = calcEntropy();

      double sumFit = 0, maxFit = -INFINITY, minFit = INFINITY;
      for (auto& ind : population) {
        double f = ind.getAccuracy();
        sumFit += f;
        if (f > maxFit) maxFit = f;
        if (f < minFit) minFit = f;
      }

      vector<SgaIndividual<N>> popCopy = population;
      vector<vector<SgaIndividual<N>>> niches;
      while (!popCopy.empty()) {
        auto bestIt = std::max_element(popCopy.begin(), popCopy.end(),
            [](const SgaIndividual<N>& a, const SgaIndividual<N>& b) {
            return a.getAccuracy() < b.getAccuracy();
            });
        vector<SgaIndividual<N>> niche;
        SgaIndividual<N> best = *bestIt;
        popCopy.erase(bestIt);
        niche.push_back(best);
        auto it = popCopy.begin();
        while (it != popCopy.end()) {
          if ((best.getGene() ^ it->getGene()).count() <= 1) {
            niche.push_back(*it);
            it = popCopy.erase(it);
          } else {
            ++it;
          }
        }
        niches.push_back(niche);
      }

      size_t largestNiche = 0;
      double bestNicheFit = -INFINITY;
      for (auto& niche : niches) {
        if (niche.size() > largestNiche) largestNiche = niche.size();
        if (niche[0].getAccuracy() > bestNicheFit) bestNicheFit = niche[0].getAccuracy();
      }

      const string dir_prefix = outputDir_.empty() ? "" : (outputDir_ + "/");
      std::ofstream stats(dir_prefix + "stats_" + name + ".csv",
                          generation == 0 ? std::ios::trunc : std::ios::app);
      if (generation == 0) {
        stats << "generation,avg_hamming,unique_genotypes,entropy,max_fitness,min_fitness,avg_fitness,num_niches,largest_niche,best_niche_fitness" << std::endl;
      }
      stats << generation << ","
        << avgDist << ","
        << unique.size() << ","
        << entropy << ","
        << maxFit << ","
        << minFit << ","
        << sumFit / population.size() << ","
        << niches.size() << ","
        << largestNiche << ","
        << bestNicheFit << std::endl;
      stats.close();

      std::error_code ec;
      const string anim_dir = dir_prefix + "anim";
      std::filesystem::create_directories(anim_dir, ec);
      std::ofstream genPop(anim_dir + "/" + name + "_gen_" + std::to_string(generation) + ".csv");
      genPop << "id,fitness\n";
      for (auto& ind : population) {
        genPop << ind.getGene().to_ulong() << "," << ind.getFitness() << "\n";
      }
    }

    double calcEntropy() {
      std::map<size_t, int> counts;
      for (auto& ind : population) {
        counts[ind.getGene().to_ulong()]++;
      }
      double entropy = 0.0;
      for (auto& [gene, count] : counts) {
        double p = (double)count / population.size();
        if (p > 0) {
          entropy -= p * std::log2(p);
        }
      }
      return entropy;
    }

    void crossover(SgaIndividual<N>& p1, SgaIndividual<N>& p2, SgaIndividual<N>& c1, SgaIndividual<N>& c2, int n) {
      uniform_int_distribution<size_t> randInt(1, N-1);
      n = std::min(n, (int)(N-1));

      set<int> pointSet;
      while ((int)pointSet.size() < n) {
        pointSet.insert(randInt(gen));
      }

      vector<int> points(pointSet.begin(), pointSet.end()); // already sorted

      vector<SgaIndividual<N>> parents = {p1, p2};
      bitset<N> cGene1, cGene2;

      size_t j = 0;
      size_t parentIndex = 0;
      for (auto crossover : points) {
        while (j < (size_t)crossover) {
          cGene1.set(j, parents[parentIndex % 2].getGene()[j]);
          cGene2.set(j, parents[(parentIndex + 1) % 2].getGene()[j]);
          j++;
        }
        parentIndex++;
      }
      while (j < N) {
        cGene1.set(j, parents[parentIndex % 2].getGene()[j]);
        cGene2.set(j, parents[(parentIndex + 1) % 2].getGene()[j]);
        j++;
      }

      c1.setGene(cGene1);
      c2.setGene(cGene2);
    }

    void init() {
      for (int i = 0; i < this->popSize; i++) {
        bitset<N> gene = this->randomGene();
        population.push_back(SgaIndividual<N>(gene, landscape->fitness(gene)));
      }
    }

    void chooseElites(vector<SgaIndividual<N>> &newPop) {
      sort(this->population.begin(), this->population.end(), 
          [](SgaIndividual<N> a, SgaIndividual<N> b){
          return a.getFitness() < b.getFitness();
          });
      newPop.insert(
          newPop.end(), 
          this->population.end()-this->numberOfElites, 
          this->population.end()
          );
    }

    void crowdingSelection(SgaIndividual<N> p1, SgaIndividual<N> p2, SgaIndividual<N> c1, SgaIndividual<N> c2, vector<SgaIndividual<N>> &res) {
      uniform_real_distribution<double> randDouble(0.0, 1.0);
      // Group by Hamming distance
      // Compete with parents
      // Find winners based on prob or det crowding
      SgaIndividual<N> o1;
      SgaIndividual<N> o2;

      if (hammingDist(p1, c1) + hammingDist(p2, c2) <= hammingDist(p1, c2) + hammingDist(p2, c1)) {
        // (P1, C1) and (P2, C2) are the optimal pairings that minimize dist
        o1 = c1;
        o2 = c2;
      } else {
        // (P1, C2) & (P2, C1)
        o1 = c2;
        o2 = c1;
      }
      auto winner = randDouble(gen) < (static_cast<double>(o1.getFitness())/(o1.getFitness() + p1.getFitness())) ? o1 : p1;
      res.push_back(winner);

      auto winner2 = randDouble(gen) < (static_cast<double>(o2.getFitness()) / (o2.getFitness() + p2.getFitness())) ? o2 : p2;
      res.push_back(winner2); 
    }

    SgaIndividual<N> run(bool verbose = true) {
      if (verbose) {
        cout << "Running algorithm..." << endl;
        if (!outputDir_.empty()) {
          std::error_code ec;
          std::filesystem::create_directories(outputDir_, ec);
          std::filesystem::create_directories(outputDir_ + "/anim", ec);
        }
      }
      uniform_int_distribution<size_t> dist(0, this->popSize-1);
      uniform_real_distribution<double> randDouble(0.0, 1.0);

      // Init solution random to avoid nullptr issues
      auto temp = this->randomGene();
      SgaIndividual<N> solution(temp);
      solution.setFitness(landscape->fitness(temp));

      for (int i = 0; i < this->generations; i++) {
        vector<SgaIndividual<N>> newPop;

        // Elite selection
        int newPopSize = this->numberOfElites;
        this->chooseElites(newPop);

        while(newPopSize < this->popSize) {
          auto tournamentSelect = [&]() -> size_t {
            size_t best = dist(gen);
            for (size_t j = 1; j < this->kParents; ++j) {
              size_t idx = dist(gen);
              if (this->population[idx].getFitness() > this->population[best].getFitness()) {
                best = idx;
              }
            }
            return best;
          };

          size_t best = tournamentSelect();
          size_t second = tournamentSelect();

          // Crossover to create 2 kids
          SgaIndividual<N> c1;
          SgaIndividual<N> c2;
          auto p1 = this->population[best];
          auto p2 = this->population[second];
          if (randDouble(gen) < this->crossoverRate) {
            crossover(p1, p2, c1, c2, this->crossoverPoints);
          } else {
            // If there's no crossover, we just copy the parents to the children.
            c1.setGene(p1.getGene());
            c2.setGene(p2.getGene());
          }

          // Mutate kids and compute fitness
          c1.mutate(this->mutationRate);
          c2.mutate(this->mutationRate);
          c1.setFitness(landscape->fitness(c1.getGene()));
          c2.setFitness(landscape->fitness(c2.getGene()));

          vector<SgaIndividual<N>> winners;
          this->crowdingSelection(p1, p2, c1, c2, winners);

          newPop.insert(newPop.end(), winners.begin(), winners.end());
          newPopSize += 2;
        }
        if (newPop.size() != this->popSize) {
          cout << "Invalid newPop size! " << newPop.size() << endl;
        }
        this->population = std::move(newPop);

        // Print average and max fitness
        double sum = 0;
        double maxFitness = -INFINITY;
        double minFitness = INFINITY;

        for (size_t j = 0; j < this->population.size(); j++) {
          double fitness = this->population[j].getFitness();
          if (fitness > maxFitness) maxFitness = fitness;
          if (fitness < minFitness) minFitness = fitness;
          sum += fitness;
        }
        if (verbose) {
          cout << "Generation: " << i 
            << " | Max fitness: " << maxFitness 
            << " | Min fitness: " << minFitness 
            << " | Average fitness: " << static_cast<double>(sum) / this->popSize << endl;
          this->dumpGeneration(i, this->name);
        }

        // Check for a new best solution
        if (maxFitness >= solution.getFitness()) {
          for (auto const &c : this->population) {
            if (c.getFitness() == maxFitness) {
              solution = SgaIndividual<N>(c.getGene());
              solution.setFitness(landscape->fitness(c.getGene()));
            }
          }
        }
      }
      return solution;
    }

    vector<vector<SgaIndividual<N>>> findNiches() {
      vector<SgaIndividual<N>> popCopy = population;
      vector<vector<SgaIndividual<N>>> niches;

      while (!popCopy.empty()) {
        auto bestIt = std::max_element(popCopy.begin(), popCopy.end(),
            [](const SgaIndividual<N>& a, const SgaIndividual<N>& b) {
            return a.getFitness() < b.getFitness();
            });

        vector<SgaIndividual<N>> niche;
        SgaIndividual<N> best = *bestIt;
        popCopy.erase(bestIt);
        niche.push_back(best);

        auto it = popCopy.begin();
        while (it != popCopy.end()) {
          if ((best.getGene() ^ it->getGene()).count() <= 1) {
            niche.push_back(*it);
            it = popCopy.erase(it);
          } else {
            ++it;
          }
        }

        niches.push_back(niche);
      }

      return niches;
    }

};
