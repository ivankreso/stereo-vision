#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>

#include "calib_helper.h"

int main(int ac, char** av) {
  if (ac != 3) {
    std::cout << av[0] << " dir_path output.xml\n";
    return 1;
  }
  std::string dir_path = av[1];
  std::string output_name = av[2];

  std::vector<std::string> files;
  CalibHelper::GetFilesInFolder(dir_path, files);
  cv::FileStorage fs(output_name, cv::FileStorage::WRITE);
  fs << "images" << "[";
  for(auto& fname : files)
    fs << fname;
  fs << "]";
  return 0;
}
