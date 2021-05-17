#include <core/accuracy.h>
#include <core/classifier.h>
#include <core/image.h>
#include <core/imagedata.h>
#include <core/model.h>

using namespace naivebayes;

int main(int argc, char* argv[]) {
  argv++;
  bool is_saving = false;
  bool is_testing = false;
  string training_labels;
  string training_images;
  string model_file;
  string classify_labels;
  string classify_images;
  if (strcmp(*argv++, "train") == 0) {  // using == leaves an error :(
    training_labels = *(argv++);
    training_images = *(argv++);
  } else {
    throw std::invalid_argument("Must provide training files");
  }
  if (strcmp(*argv++, "save") == 0) {
    model_file = *(argv);
    is_saving = true;
  }

  if (strcmp(*argv++, "test") == 0) {
    classify_labels = *(argv++);
    classify_images = *(argv);
    is_testing = true;
  }

  ImageData training_data(training_labels, training_images);
  Model model(training_data);

  if (is_saving)
    model.PrintModel(model_file);

  if (is_testing) {
    Accuracy accuracy;
    accuracy.GetAccuracy(model, training_data, training_labels,
                         training_labels);
  }

  return 0;
}
