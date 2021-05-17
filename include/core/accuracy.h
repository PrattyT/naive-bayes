//
// Created by Pratyush Tulsian on 10/20/20.
//

#pragma once

#include "model.h"
namespace naivebayes {

class Accuracy {

 public:

  /**
   * @return a double containing the accuracy of a model to classify a file
   * of images.
   */
  double GetAccuracy(Model& model, ImageData& image_data, string& classify_labels,
                     string& classify_images);
};

}  // namespace naivebayes