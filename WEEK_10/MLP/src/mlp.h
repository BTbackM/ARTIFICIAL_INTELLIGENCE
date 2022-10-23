#pragma once

#include <armadillo>
#include <vector>

using namespace arma;
using namespace std;

struct MLP {
  int N;
  vector<Mat<double>> weights;

  MLP(const int N);
  
  MLP(const int N, const int H, const int O);

  void backward(Mat<double> input);

  void forward(Mat<double> input);

  void train();
  
  ~MLP();
};
