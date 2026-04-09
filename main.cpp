#include "landscapes.cpp"
#include <iostream>

using namespace std;

int main() {
  TriangleLandscape<16> test(1, 4);
  bitset<16> t(0);
  for (int i = 0; i < 16; i++) {
    cout << test.fitness(t).first << endl;
    t.set(i, true);
  }

  return 0;
}
