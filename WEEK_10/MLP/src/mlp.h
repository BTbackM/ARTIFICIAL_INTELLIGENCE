#pragma once

#include <armadillo>
#include <memory>
#include <vector>

using namespace arma;
using namespace std;

#include "activation.h"

struct MLP {
  int N;
  shared_ptr<Function> f_ac;
  vector<Mat<double>> weights;
  vector<Mat<double>> outputs;

  MLP(shared_ptr<Function> f_ac, const int N);
  
  MLP(shared_ptr<Function> f_ac, const int N, const int H, const int O);

  void backward(Mat<double> input, const int alpha);

  void forward(Mat<double> input);

  void train(Mat<double> train_set, const int epochs);
  
  void predict(Mat<double> predict_set);

  ~MLP();
};
