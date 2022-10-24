#include <armadillo>
#include <iostream>
#include <memory>

using namespace arma;

#include "activation.h"
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

  shared_ptr<Function> f_ac = make_shared<Sigmoid>();
  
  MLP mlp = MLP(f_ac, N, H, O);
  Mat<double> input({1, 2, 3});
  mlp.forward(input);
  double alpha = 0.01;
  Mat<double> output({0.5, 0.5, 0.5});
  mlp.backward(output, alpha);

  return 0;
}
