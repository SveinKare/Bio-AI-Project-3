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
