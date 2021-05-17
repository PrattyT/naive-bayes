//
// Created by Pratyush Tulsian on 10/11/20.
//

#include <core/model.h>

#include <fstream>
#include <vector>

using std::endl;
using std::ifstream;
using std::ofstream;
using std::stod;
using std::to_string;
using std::vector;
namespace naivebayes {

Model::Model(ImageData& training_data) {
  class_probability_ = vector<double>(kClasses);

  CalculatePixelProbabilities(training_data);
  CalculateClassProbabilities(training_data);
}

Model::Model(string& file_name) {
  ifstream myFile;
  myFile.open(file_name);
  string line;
  getline(myFile, line);
  image_size_ = stod(line);

  // create new probabilities map and class probabilities vector
  for (size_t label = 0; label < kClasses; label++)
    probabilities_[label] =
        vector<vector<double>>(image_size_, vector<double>(image_size_, 0));

  class_probability_ = vector<double>(kClasses);

  // read class probabilities
  if (myFile.good())
    myFile >> class_probability_;

  // add each class' probabilities for each pixel
  if (myFile.good())
    myFile >> probabilities_;
  image_size_ = probabilities_[0][0].size();
  myFile.close();
}

istream& operator>>(istream& is, map<int, vector<vector<double>>>& data) {
  string line;
  size_t image_size = data[0][0].size();

  // pull in each pixel
  for (size_t c = 0; c < kClasses; c++)
    for (size_t row = 0; row < image_size; row++)
      for (size_t col = 0; col < image_size; col++) {
        getline(is, line);
        data[c][row][col] = stod(line);
      }

  return is;
}

istream& operator>>(istream& is, vector<double>& data) {
  string line;
  for (size_t c = 0; c < kClasses; c++) {
    getline(is, line);
    data[c] = stod(line);
  }
  return is;
}

// operator overload directly into model
//istream& operator>>(istream& is, Model& model) {
//  map<int, vector<vector<double>>> data = model.GetProbabilities();

//  string line;
//  getline(is, line);
//  image_size_ = stod(line);
//
//  // read class probabilities.
//  for (size_t c = 0; c < kClasses; c++) {
//    getline(is, line);
//    data[c] = stod(line);
//  }
//
//  size_t image_size = data[0][0].size();
//
//  // pull in each pixel probability
//  for (size_t c = 0; c < kClasses; c++)
//    for (size_t row = 0; row < image_size; row++)
//      for (size_t col = 0; col < image_size; col++) {
//        getline(is, line);
//        data[c][row][col] = stod(line);
//      }
//
//  return is;
//}

void Model::PrintModel(string file_name) {
  ofstream outfile(file_name);
  outfile << image_size_ << endl;

  // print class probabilities
  for (double probability : class_probability_)
    outfile << probability << endl;

  // print probabilities at each pixel
  for (size_t c = 0; c < kClasses; c++) {
    PrintProbability(outfile, probabilities_[c]);
  }
}

void Model::PrintProbability(ostream& outfile,
                             vector<vector<double>>& probabilities) {
  for (auto& v : probabilities)
    for (double probability : v)
      outfile << probability << endl;
}

double Model::ComputeShadedProbability(size_t row, size_t col,
                                       vector<Image> images) {
  double validImages = 0;
  double count = 0;
  for (Image image : images)
    if (image.GetImageSize() != 0) {
      validImages++;
      if (image.IsShaded(row, col))
        count++;
    }

  // apply laplace smoothing
  return (kLaplace + count) / (2.0 * kLaplace + validImages);
}

map<int, vector<vector<double>>> Model::GetProbabilities() {
  return probabilities_;
}
vector<double> Model::GetClassProbabilites() {
  return class_probability_;
}
void Model::CalculatePixelProbabilities(ImageData& training_data) {
  map<int, vector<Image>> data = training_data.GetTrainingData();
  image_size_ = data[0][0].GetImageSize();

  // Calculate shaded probabilities per image
  for (size_t label = 0; label < kClasses; label++) {
    vector<vector<double>> class_probability =
        vector<vector<double>>(image_size_, vector<double>(image_size_, 0));
    vector<Image> images = data[label];

    // calculate probabilities per pixel
    for (size_t row = 0; row < image_size_; row++) {
      for (size_t col = 0; col < image_size_; col++) {
        class_probability[row][col] =
            ComputeShadedProbability(row, col, images);
      }
    }
    probabilities_[label] = class_probability;
  }
}
void Model::CalculateClassProbabilities(ImageData& training_data) {
  map<int, vector<Image>> data = training_data.GetTrainingData();
  for (size_t c = 0; c < kClasses; c++) {
    // apply laplace smoothing
    class_probability_[c] =
        (kLaplace + data[c].size()) /
        (training_data.GetTrainingImagesSize() + 2.0 * kLaplace);
  }
}

}  // namespace naivebayes
