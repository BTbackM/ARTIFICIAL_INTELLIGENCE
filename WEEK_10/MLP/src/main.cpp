#include <armadillo>
#include <Eigen/Eigen>
#include <iostream>

using namespace arma;

#include "mlp.h"

int main(int argc, char *argv[]) {
  int N, H, O;

  if(argc == 4) {
    N = atoi(argv[1]);
    H = atoi(argv[2]);
    O = atoi(argv[3]);
  }
  else {
    N = 3;
    H = 4;
    O = 3;
  }

  MLP mlp = MLP(N, H, O);
  Mat<double> input({1, 2, 3});
  mlp.forward(input);

  return 0;
}
