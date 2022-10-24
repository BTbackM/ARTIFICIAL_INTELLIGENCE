#include "activation.h"

Mat<double> Sigmoid::calculate(Mat<double> input) {
  Mat<double> sigmoid = (1.0) / (1.0 + exp(-input));

  return sigmoid;
}

Mat<double> Sigmoid::derivate(Mat<double> input) {
  Mat<double> output = calculate(input); 

  return output * (1.0 - output);
}
