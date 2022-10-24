#include <iostream>

#include "mlp.h"

MLP::MLP(shared_ptr<Function> f_ac, const int N) {
  this->f_ac = f_ac;
  this->N = N;

  Mat<double> output = Mat<double>(N + 1, 1, fill::ones);
  for(int i = 0; i < N; i++) {
    Mat<double> weight = Mat<double>(N + 1, N, fill::randu);

    this->weights.push_back(weight);
    this->outputs.push_back(output);
  }
  this->outputs.push_back(output);
}

MLP::MLP(shared_ptr<Function> f_ac, const int N, const int H, const int O) {
  this->f_ac = f_ac;
  this->N = N;

  Mat<double> output;
  for(int i = 0; i < this->N; i++) {
    Mat<double> weight;

    if(i == 0) {
      weight = Mat<double>(N + 1, H, fill::randu);
      output = Mat<double>(N + 1, 1, fill::ones);
    }
    else if(i == (N - 1)) {
      weight = Mat<double>(H + 1, O, fill::randu);
      output = Mat<double>(H + 1, 1, fill::ones);
    }
    else {
      weight = Mat<double>(H + 1, H, fill::randu);
      output = Mat<double>(H + 1, 1, fill::ones);
    }

    this->weights.push_back(weight);
    this->outputs.push_back(output);
  }
  output = Mat<double>(O, 1, fill::ones);
  this->outputs.push_back(output);
}

void MLP::backward(Mat<double> S_d, const double alpha) {
  // NOTE: Hiden - Output back propagation
  Mat<double> S_o = this->outputs[this->N];
  Mat<double> S_i = this->outputs[this->N - 1];
  S_i = join_rows(mat({1}), S_i);

  this->lambda = (S_o - S_d) % (S_o % (1.0 - S_o));
  Mat<double> derivate = this->lambda.t() * S_i;
  this->weights[this->N - 1] -= (alpha * derivate.t());

  // NOTE: Hiden - Hiden back propagation
  double sum_lambda = accu(this->lambda);
  for(int i = (this->N - 1); i > 0; i--) {
    S_o = this->outputs[i];
    S_i = this->outputs[i - 1];
    S_i = join_rows(mat({1}), S_i);

    derivate = (sum_lambda * (S_o % (1.0 - S_o))).t() * S_i;
    this->weights[i - 1] -= (alpha * derivate.t());
  }
}

void MLP::forward(Mat<double> S_i) {
  this->outputs[0] = S_i;
  Mat<double> S = join_rows(mat({1}), S_i);

  for(int i = 0; i < this->N; i++) {
    S = S * weights[i];
    S = f_ac->calculate(S);
    this->outputs[i + 1] = S;
    if(i != (N - 1))
      S = join_rows(mat({1}), S);
  }
}

void MLP::train(Mat<double> train_set, const int epochs) {
  train_set.print("Train:");
  for(int i = 0; i < epochs; i++) {

  }
}

MLP::~MLP() {
}
