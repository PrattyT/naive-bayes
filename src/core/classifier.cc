//
// Created by Pratyush Tulsian on 10/18/20.
//

#include <core/classifier.h>
#include <core/imagedata.h>

namespace naivebayes {

Classifier::Classifier(Model& model) {
  probabilities_ = model.GetProbabilities();
  class_probabilities_ = model.GetClassProbabilites();
}

vector<int> Classifier::ClassifyFromFile(string& image_file,
                                         string& label_file) {
  vector<int> classes;
  vector<Image> images = createImages(image_file, label_file);

  // classify each image
  for (Image image : images)
    classes.push_back(ClassifyImage(image));

  return classes;
}

int Classifier::ClassifyImage(Image& image) {
  proabability_by_class_ = vector<double>();

  for (int i = 0; i < kClasses; i++)
    proabability_by_class_.push_back(ProbabilityOfClass(i, image));

  // find max probability
  int best_class = 0;
  double best_probability = proabability_by_class_[0];
  for (int i = 0; i <kClasses; i++)
    if (proabability_by_class_[i] > best_probability) {
      best_probability = proabability_by_class_[i];
      best_class = i;
    }

  return best_class;
}

vector<Image> Classifier::createImages(string& label_file, string& image_file) {
  ImageData image_data(image_file);
  return image_data.GetImages();
}

double Classifier::ProbabilityOfClass(int c, Image& image) {
  double sum_probabilities = log10(class_probabilities_[c]);

  for (int row = 0; row < image.GetImageSize(); row++)
    for (int col = 0; col < image.GetImageSize(); col++)
      sum_probabilities += GetProbability(image, c, row, col);

  return sum_probabilities;
}

double Classifier::GetProbability(Image& image, int c, int row, int col) {
  if (image.IsShaded(row, col))
    return log10(probabilities_[c][row][col]);
  else
    return log10(1.0 - probabilities_[c][row][col]);
}
vector<double> Classifier::GetProbabilityByClass() {
  return proabability_by_class_;
}

}  // namespace naivebayes