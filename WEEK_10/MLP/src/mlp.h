#pragma once

#include <armadillo>
#include <memory>
#include <vector>

using namespace arma;
using namespace std;

#include "activation.h"

struct MLP {
  int N;
  Mat<double> lambda;
  shared_ptr<Function> f_ac;
  vector<Mat<double>> outputs;
  vector<Mat<double>> weights;

  MLP(shared_ptr<Function> f_ac, const int N);
  
  MLP(shared_ptr<Function> f_ac, const int N, const int H, const int O);

  void backward(Mat<double> input, const double alpha);

  void forward(Mat<double> input);

  vector<Mat<double>> predict(vector<Mat<double>> input);

  void train(vector<Mat<double>> input, vector<Mat<double>> output, const int epochs, const int alpha);

  ~MLP();
};
