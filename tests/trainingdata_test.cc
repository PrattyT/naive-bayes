#include <core/image.h>
#include <core/imagedata.h>
#include <core/model.h>

#include <catch2/catch.hpp>
#include <string>
#include <vector>

using namespace naivebayes;
using std::string;
using std::vector;

string labels =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/"
    "naivebayes-Proctu/tests/traininglabelstest2";
string images =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/"
    "naivebayes-Proctu/tests/trainingimagestest2";

TEST_CASE("Check reading in valid input") {
  ImageData training_data(kLabels, kImages);
  map<int, vector<Image>> data = training_data.GetTrainingData();
  SECTION("Check Lables file") {
    vector<int> expected = {0, 1, 1};
    REQUIRE(training_data.GetLabelData() == expected);
  }
  SECTION("Check first image") {
    vector<vector<bool>> expected = {
        {true, true, true}, {true, false, true}, {true, true, true}};
    REQUIRE(data[0][0].GetImageData() == expected);
  }
  SECTION("Check last image") {
    vector<vector<bool>> expected = {
        {false, true, false}, {false, true, false}, {false, true, false}};
  }
  SECTION("Different Size image") {
    ImageData smaller_images(labels, images);
    vector<vector<bool>> expected = {{true, true}, {false, true}};
    REQUIRE(smaller_images.GetTrainingData()[0][0].GetImageData() == expected);
  }
}

TEST_CASE("Bad file arguments") {
  string labels_bad =
      "/Users/pratyushtulsian/Documents/~Cinder/my-projects/"
      "naivebayes-Proctu/tests/trai";
  string images_bad =
      "/Users/pratyushtulsian/Documents/~Cinder/my-projects/"
      "naivebayes-Proctu/tests/trai";
  REQUIRE_THROWS_AS(ImageData(labels_bad, images_bad),
                    std::invalid_argument);
}
