//
// Created by Pratyush Tulsian on 10/18/20.
//

#include <core/accuracy.h>
#include <core/classifier.h>
#include <core/imagedata.h>
#include <core/model.h>

#include <catch2/catch.hpp>

using namespace naivebayes;

string images_file_train =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "data/trainingimages";
string labels_file_train =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "data/traininglabels";

string images_file_classify =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "data/testimages";
string labels_file_classify =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "data/testlabels";

TEST_CASE("Classify individual image", "[size =3]") {
  ImageData training_data(kLabels, kImages);
  Model model(training_data);
  Classifier classifier(model);
  Image image(3);
  string row_one = "## ";
  string row_two = " # ";
  string row_three = " # ";
  image.AddRow(0, row_one);
  image.AddRow(1, row_two);
  image.AddRow(2, row_three);
  int classification = classifier.ClassifyImage(image);
  vector<double> probability_by_class = classifier.GetProbabilityByClass();
  SECTION("Check classification") {
    REQUIRE(classification == 1);
  }
  SECTION("Check Mathematical correctness for a 0") {
    REQUIRE(probability_by_class[0] == Approx(-3.7889414));
  }
  SECTION("Check Mathematical correctness for a 1") {
    REQUIRE(probability_by_class[1] == Approx(-1.874571155));
  }
  SECTION("Check correctness of a class with no training data") {
    REQUIRE(probability_by_class[2] ==
            Approx(-3.408239965));  // log(0.2 x 0.5^9)
  }
}

TEST_CASE("Classify individual image different size", "[size =2]") {
  // file paths for images of size 2
  string labels_two =
      "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
      "tests/traininglabelstest2";

  string image_two =
      "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
      "tests/trainingimagestest2";

  ImageData training_data(labels_two, image_two);
  Model model(training_data);
  Classifier classifier(model);
  Image image(2);
  string row_one = "## ";
  string row_two = " # ";
  image.AddRow(0, row_one);
  image.AddRow(1, row_two);
  int classification = classifier.ClassifyImage(image);
  vector<double> probability_by_class = classifier.GetProbabilityByClass();
  SECTION("Check classification") {
    REQUIRE(classification == 0);
  }
  SECTION("Check Mathematical correctness for a 0") {
    REQUIRE(probability_by_class[0] == Approx(-0.8804562));
  }
  SECTION("Check Mathematical correctness for a 1") {
    REQUIRE(probability_by_class[1] == Approx(-1.6812412));
  }
}

TEST_CASE("Classify images in a file") {
  ImageData training_data(labels_file_train, images_file_train);
  string file_name =
      "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
      "tests/trainingimages5000";
  Model model(file_name);
  Classifier classifier(model);
  vector<int> expected = training_data.GetLabelData();

  SECTION("Check size of vectors") {
    REQUIRE(5000 == expected.size());
  }
  SECTION("Check accuracy") {
    Accuracy accuracy;
    REQUIRE(accuracy.GetAccuracy(model, training_data, labels_file_classify,
                                 images_file_classify) > 0.7);
  }
}