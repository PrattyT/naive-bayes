#include <core/imagedata.h>
#include <core/model.h>

#include <catch2/catch.hpp>
#include <string>
#include <vector>

using namespace naivebayes;
using std::string;
using std::vector;

ImageData training_data(kLabels, kImages);
string file_name =
    "/Users/pratyushtulsian/Documents/~Cinder/my-projects/naivebayes-Proctu/"
    "tests/data";
Model model(training_data);

TEST_CASE("Test Probabilities per pixel from training data") {
  map<int, vector<vector<double>>> probabilities = model.GetProbabilities();
  SECTION("Test First class probabilities") {
    vector<vector<double>> expected = {{2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0},
                                       {2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0},
                                       {2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}};
    REQUIRE(probabilities[0] == expected);
  }
  SECTION("Test Last class probabilities") {
    vector<vector<double>> expected = {{2.0 / 4.0, 3.0 / 4.0, 1.0 / 4.0},
                                       {1.0 / 4.0, 3.0 / 4.0, 1.0 / 4.0},
                                       {2.0 / 4.0, 3.0 / 4.0, 2.0 / 4.0}};
    REQUIRE(probabilities[1] == expected);
  }
  SECTION("Test class with no data") {
    vector<vector<double>> expected = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}};
    REQUIRE(probabilities[9] == expected);
  }
}

TEST_CASE("Test Class probabilities") {
  vector<double> expected = {0.4, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
  vector<double> modelProbabilities = model.GetClassProbabilites();
  for (size_t i = 0; i < kClasses; i++)
    REQUIRE(expected[i] == Approx(modelProbabilities[i]));
}

TEST_CASE("Writing a model to a file") {
  model.PrintModel(file_name);
  ifstream myFile;
  myFile.open(file_name);
  string line;
  size_t count = 0;
  while (getline(myFile, line)) count++;
  size_t image_size = model.GetProbabilities()[0][0].size();
  REQUIRE(count == 1 + kClasses + image_size*image_size * kClasses);
}

TEST_CASE("Creating a model from a file") {
  model.PrintModel(file_name);
  Model model_from_file(file_name);
  SECTION("Check individual pixels") {
    for (size_t c = 0; c < kClasses; c++)
      for (size_t row = 0; row < model.GetProbabilities()[0].size(); row++)
        for (size_t col = 0; col < model.GetProbabilities()[0].size(); col++)
          REQUIRE(model_from_file.GetProbabilities()[c][row][col] ==
                  Approx(model.GetProbabilities()[c][row][col]));
  }
  SECTION("Check class probabilities") {
    vector<double> expected = {0.4, 0.6, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2};
    for (size_t i = 0; i < kClasses; i++)
      REQUIRE(expected[i] == Approx(model_from_file.GetClassProbabilites()[i]));
  }
}