#include <bitset>

using namespace std;

template<size_t N>
class Landscape {
  public:
    virtual double fitness(bitset<N> gene) const = 0;
    virtual ~Landscape() = default;
};

template<size_t N>
class TriangleLandscape : public Landscape<N> {
  private:
    int n;
    int m;
    int s;
  public:
    TriangleLandscape(int n, int m, int s): n(n), m(m), s(s) {};
    double fitness(bitset<N> gene);
    int triangle(bitset<N> gene);
    int g(size_t b);
};
