//
// Created by Pratyush Tulsian on 10/11/20.
//

#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "image.h"
#include "imagedata.h"

using std::ifstream;
using std::map;
using std::ostream;
using std::vector;

namespace naivebayes {
/**
 * Represents a model for implementing a naive bayes classification.
 */
class Model {
 public:
  /**
   * Create a new Model with training data.
   */
  Model(ImageData &training_data);

  /**
   * Load a model from a file
   */
  Model(string &file_name);

  /**
   * Write the model to a file
   */
  void PrintModel(string file_name);

  /**
   * @return a mapping from each classification to a 2D vector of probabilities
   * for each pixel to be shaded.
   */
  map<int, vector<vector<double>>> GetProbabilities();

  /**
   * @return a vector containing the probabilities for each class.
   */
  vector<double> GetClassProbabilites();

 private:
  // laplace smoothing k
  const double kLaplace = 1.0;

  size_t image_size_;

  // stores the probabilities of each pixel being filled for each class.
  map<int, vector<vector<double>>> probabilities_;

  // class probability
  vector<double> class_probability_;

  void CalculatePixelProbabilities(ImageData &trainingData);
  void CalculateClassProbabilities(ImageData &trainingData);

  // Calculate the probability a pixel is shaded for all images in a class.
  double ComputeShadedProbability(size_t row, size_t col, vector<Image> images);

  // print the probabilities for a class to a file
  void PrintProbability(ostream &outfile,
                        vector<vector<double>> &probabilities);
};

/**
 * Overload the >> operator to read probabilities from a file into a map.
 */
istream &operator>>(istream &is, map<int, vector<vector<double>>> &data);

/**
 * Overload the >> operator to read class probabilities from a file into a
 * vector.
 */
istream &operator>>(istream &is, vector<double> &data);

/**
 * Overload the >> operator to read probabilities from a file into a map.
 */
istream &operator>>(istream &is, Model &model);

}  // namespace naivebayes
