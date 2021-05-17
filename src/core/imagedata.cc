#include <core/image.h>
#include <core/imagedata.h>

#include <fstream>
#include <iostream>
#include <string>

using std::ifstream;
using std::invalid_argument;
using std::stoi;
using std::string;
namespace naivebayes {

ImageData::ImageData(const string& labels_file, const string& images_file) {
  labels_file_ = labels_file;
  images_file_ = images_file;
  ReadTrainingLabels();

  // initialize training data map
  for (size_t label = 0; label < kClasses; label++)
    training_data_[label] = vector<Image>();

  ReadTrainingImages();
}

ImageData::ImageData(const string& images_file) {
  images_file_ = images_file;
  ifstream myFile(images_file_);
  if (myFile.fail())
    throw invalid_argument("Bad file");
  while (myFile.good()) {
    // push each image into the vector.
    myFile >> images_;
  }
  myFile.close();
}

void ImageData::ReadTrainingLabels() {
  ifstream myFile;
  myFile.open(labels_file_);
  if (myFile.fail())
    throw invalid_argument("Bad file");
  myFile >> labels_;
  myFile.close();

  int position = 0;
  for (Image& image : images_) {
    image.SetLabel(labels_[position]);
  }
}

istream& operator>>(istream& is, vector<int>& vector) {
  string line;
  while (getline(is, line))
    vector.push_back(stoi(line));
  return is;
}

void ImageData::ReadTrainingImages() {
  // read images in order
  ifstream myFile(images_file_);
  if (myFile.fail())
    throw invalid_argument("Bad file");
  while (myFile.good()) {
    // push each image into the vector.
    myFile >> images_;
  }
  myFile.close();

  // read images to store in map
  size_t position = 0;
  ifstream myFile2(images_file_);
  if (myFile2.fail())
    throw invalid_argument("Bad file");
  while (myFile2.good()) {
    // push each image into the map.
    myFile2 >> training_data_[labels_[position++]];
  }
  myFile2.close();
}

istream& operator>>(istream& is, vector<Image>& vector) {
  string line;
  getline(is, line);

  // get image size and create a new image
  size_t image_size_ = line.size();
  Image image = Image(image_size_);
  image.AddRow(0, line);

  // populate the image
  for (size_t row = 1; row < image_size_; row++) {
    getline(is, line);
    image.AddRow(row, line);
  }

  vector.push_back(image);
  return is;
}

// operator overload directly into imagedata
//istream& operator>>(istream& is, ImageData& image_data) {
//  string line;
//  getline(is, line);
//  map<int, vector<Image>> vector;
//  size_t position = 0;
//  while (is.good()) {
//    // get image size and create a new image
//    size_t image_size_ = line.size();
//    Image image = Image(image_size_);
//    image.SetLabel(labels[position++];
//    image.AddRow(0, line);
//
//    // populate the image
//    for (size_t row = 1; row < image_size_; row++) {
//      getline(is, line);
//      image.AddRow(row, line);
//    }
//
//    vector.push_back(image);
//  }
//
//  return is;
//}

map<int, vector<Image>> ImageData::GetTrainingData() {
  return training_data_;
}
vector<int> ImageData::GetLabelData() {
  return labels_;
}
size_t ImageData::GetTrainingImagesSize() {
  return labels_.size();
}
vector<Image> ImageData::GetImages() {
  return images_;
}

}  // namespace naivebayes
