//
// Created by Pratyush Tulsian on 10/18/20.
//

#pragma once
#include <string>

#include "model.h"

using std::string;
namespace naivebayes {

class Classifier {
 public:
  /**
   * Create a new Classifier from a given model.
   */
  Classifier(Model& model);

  /**
   * Classify a file of images
   * @return vector of class labels corresponding to each image in the file
   */
  vector<int> ClassifyFromFile(string& image_file, string& label_file);

  /**
   * @return vector containing the probability that the most recent image is
   * of any given class.
   */
  vector<double> GetProbabilityByClass();

  /**
  * @return the best classification of an image
  */
  int ClassifyImage(Image& image);

 private:
  map<int, vector<vector<double>>> probabilities_;
  vector<double> class_probabilities_;
  vector<double> proabability_by_class_;

  /**
   * Create the vector of images to classify
   * @return vector of images
   */
  vector<Image> createImages(string& image_file, string& label_file);

  /**
   * @return the probability that the image if of class c.
   */
  double ProbabilityOfClass(int c, Image& image);

  /**
   * @return the probability that Pixel(row, col) is shaded/unshaded in class c
   */
  double GetProbability(Image& image, int c, int row, int col);
};

}  // namespace naivebayes
