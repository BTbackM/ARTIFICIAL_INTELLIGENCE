#pragma once

#include <armadillo>

using namespace arma;

struct Function {
  virtual Mat<double> calculate(Mat<double> input) = 0;
  virtual Mat<double> derivate(Mat<double> input) = 0;

  virtual ~Function() {};
};

struct Sigmoid : Function {
  Mat<double> calculate(Mat<double> input) override;
  Mat<double> derivate(Mat<double> input) override;

  ~Sigmoid() override {}
};
