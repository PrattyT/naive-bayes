//
// Created by Pratyush Tulsian on 10/10/20.
//
#pragma once

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace naivebayes {

/**
 * The image class represents an image stored as a 2D vector of booleans.
 */
class Image {

 public:
  Image(vector<bool>& image_data);
  /**
   * Is the Pixel(Row, Col) shaded?
   */
  bool IsShaded(int row, int col);

  /**
   * Append a row to the image
   */
  void AddRow(int row, string &line);

  /**
   * Create a new Image with a side length of get size.
   */
  Image(size_t get_size);

  /**
   * Create a new Image with a side length of get size.
   */
  Image(vector<vector<bool>>& data);

  /**
   * @return the side length of the image.
   */
  size_t GetImageSize();

  /**
   * @return a map of the image data in a 2D vector.
   */
  vector<vector<bool>> GetImageData();

  void SetLabel(int label);


 private:
  // true indicates shaded pixel
  vector<vector<bool>> image_data_;

  int label_;

  // side length of image
  size_t image_side_length_;
};

}  // namespace naivebayes
