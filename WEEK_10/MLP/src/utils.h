#pragma once

#include<algorithm>
#include <armadillo>
#include <fstream>
#include <string>
#include <vector>

using namespace arma;
using namespace std;

static string DATA_PATH = "data";

static vector<Mat<double>> read_data(const string path, int rows) {
  vector<Mat<double>> dataset;
  vector<string> row;
  vector<double> row_d;
  string line, word;

  fstream input(path, ios::in);
  if(input.is_open()) {
    while(getline(input, line) and --rows) {
      row.clear();

      stringstream str(line);
      while(getline(str, word, ',')) {
        row.push_back(word);
      }

      string output = row.back();
      row.pop_back();

      transform(row.begin(), row.end(), back_inserter(row_d),
        [&](string s) {
          return stof(s);
        });

      dataset.push_back(Mat<double>(row_d));
    }
  }

  return dataset;
}
