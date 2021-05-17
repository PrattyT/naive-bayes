//
// Created by Pratyush Tulsian on 10/20/20.
//

#include "core/accuracy.h"

#include <core/classifier.h>

double naivebayes::Accuracy::GetAccuracy(naivebayes::Model &model,
                                         naivebayes::ImageData &image_data,
                                         string &classify_labels,
                                         string &classify_images) {
  Classifier classifier(model);
  vector<int> classes =
      classifier.ClassifyFromFile(classify_labels, classify_images);
  vector<int> expected = image_data.GetLabelData();
  double correct = 0;
  int position = 0;

  for (int classification : expected)
    if (classification == classes[position++])
      correct++;

  return  correct / 1000.0;


}
