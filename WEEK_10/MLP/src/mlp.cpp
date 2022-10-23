#include <iostream>

#include "mlp.h"

MLP::MLP(const int N) {
  for(int i = 0; i < N; i++) {
    Mat<double> weight = Mat<double>(N, N, fill::randu);

    weights.push_back(weight);
    // weights[i].print("W:");
  }
}

MLP::MLP(const int N, const int H, const int O) {
  this->N = N;

  for(int i = 0; i < this->N; i++) {
    Mat<double> weight;

    if(i == 0)
      weight = Mat<double>(N + 1, H, fill::randu);
    else if(i == (N - 1))
      weight = Mat<double>(H + 1, O, fill::randu);
    else
      weight = Mat<double>(H + 1, H, fill::randu);

    weights.push_back(weight);
    // weights[i].print("W:");
  }
}



void MLP::forward(Mat<double> input) {
  Mat<double> S = join_rows(mat({1}), input);

  for(int i = 0; i < this->N; i++) {
    S.print("S:");
    weights[i].print("W:");
    S = S * weights[i];
    S.print("R:");
    if(i != (N - 1))
      S = join_rows(mat({1}), S);
  }

  S.print("O:");
}

void MLP::train() {}

MLP::~MLP() {}
