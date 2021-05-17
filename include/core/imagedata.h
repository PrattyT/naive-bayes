#pragma clang diagnostic push
#pragma once

#include <map>
#include <string>
#include <vector>

#include "image.h"

using std::istream;
using std::map;
using std::string;
using std::vector;

namespace naivebayes {

const size_t kClasses = 10;

// main test file paths
const string kLabels =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "tests/traininglabelstest3";
const string kImages =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "tests/trainingimagestest3";

/**
 * Represents the Training data comprised of images used to train the
 * Naive Bayes Model.
 */
class ImageData {
 public:
  /**
   * Creates a new ImageData object from a labels and image file.
   */
  ImageData(const string& labels_file, const string& images_file);

  ImageData(const string& images_file);
  /**
   * @return the training data in the form of a mapping from each class to
   * a vector of all the images that are describing that class.
   */
  map<int, vector<Image>> GetTrainingData();

  /**
   * @return a vector with all the labels.
   */
  vector<int> GetLabelData();

  size_t GetTrainingImagesSize();

  vector<Image> GetImages();

 private:
  /**
   * read in the training labels
   */
  void ReadTrainingLabels();

  /**
   * read in the training Images.
   */
  void ReadTrainingImages();

  // tracks all training labels
  vector<int> labels_;

  // all of the training data
  map<int, vector<Image>> training_data_;

  vector<Image> images_;
  // file paths for this training data
  string labels_file_;
  string images_file_;
};

/**
 * Overload the >> operator to read labels from a file into a vector.
 */
istream& operator>>(istream& is, vector<int>& vector);

/**
 * Overlaod the >> operator to read an image from a file into a vector.
 */
istream& operator>>(istream& is, vector<Image>& vector);

/**
 * Overlaod the >> operator to read a file of images into training data
 */
istream& operator>>(istream& is, ImageData& trainging_data);

}  // namespace naivebayes
