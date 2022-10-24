#include <iostream>

#include "mlp.h"

MLP::MLP(shared_ptr<Function> f_ac, const int N) {
  this->f_ac = f_ac;
  this->N = N;

  for(int i = 0; i < N; i++) {
    Mat<double> weight = Mat<double>(N + 1, N, fill::randu);
    Mat<double> output = Mat<double>(N + 1, 1, fill::ones);

    weights.push_back(weight);
    outputs.push_back(output);
    // weights[i].print("W:");
    // outputs[i].print("O:");
  }
}

MLP::MLP(shared_ptr<Function> f_ac, const int N, const int H, const int O) {
  this->f_ac = f_ac;
  this->N = N;

  for(int i = 0; i < this->N; i++) {
    Mat<double> weight;
    Mat<double> output;

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

    weights.push_back(weight);
    outputs.push_back(output);
    // weights[i].print("W:");
    // outputs[i].print("O:");
  }
}

void MLP::backward(Mat<double> output, const int alpha) {
  output.print("Print:");

  for(int i = (this->N - 1); i > 0; i--) {
    Mat<double> lambda = (output - output) % (output % (1.0 - output));
    // this->weights = this->weights - (alpha * derivate);
  }

}

void MLP::forward(Mat<double> input) {
  Mat<double> S = join_rows(mat({1}), input);

  for(int i = 0; i < this->N; i++) {
    // S.print("S:");
    // weights[i].print("W:");
    S = S * weights[i];
    S = f_ac->calculate(S);
    // S.print("R:");
    if(i != (N - 1))
      S = join_rows(mat({1}), S);
    outputs[i] = S;
    outputs[i].print("O:");
  }
}

void MLP::train(Mat<double> train_set, const int epochs) {
  train_set.print("Train:");
  for(int i = 0; i < epochs; i++) {}
}

MLP::~MLP() {
}
