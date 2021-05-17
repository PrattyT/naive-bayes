//
// Created by Pratyush Tulsian on 10/10/20.
//

#include <core/image.h>

#include <iostream>

namespace naivebayes {

bool naivebayes::Image::IsShaded(int row, int col) {
  return image_data_[row][col];
}

void Image::AddRow(int row, string &line) {
  int col = 0;
  for (char c : line) {
    image_data_[row][col++] = (c != ' ');
  }
}

Image::Image(size_t get_size) {
  image_data_ = vector<vector<bool>>(get_size, vector<bool>(get_size, 0));
  image_side_length_ = get_size;
}

vector<vector<bool>> Image::GetImageData() {
  return image_data_;
}
size_t Image::GetImageSize() {
  return image_side_length_;
}
void Image::SetLabel(int label) {
  label_ = label;
}

Image::Image(vector<vector<bool>>& data) {
  image_data_ = data;
  image_side_length_ = data.size();
}

}  // namespace naivebayes
