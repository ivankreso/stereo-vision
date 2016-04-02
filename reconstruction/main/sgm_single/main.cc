#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../base/stereo_sgm.h"

void RunSGM(const int P1, const int P2, const std::string left_img_fname, const std::string right_img_fname,
            const std::string output_folder)
{
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);

  //cv::Mat img_disp, img_disp_subpixel, mat_disp;
  cv::Mat img_disp, img8_disp;
  cv::Mat img_left = cv::imread(left_img_fname, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(right_img_fname, CV_LOAD_IMAGE_GRAYSCALE);
  //cv::imshow("img_left", img_left);
  //cv::waitKey(0);

  //double sigma = 0.7;
  //cv::GaussianBlur(img_left, img_left, cv::Size(3,3), sigma);
  //cv::GaussianBlur(img_right, img_right, cv::Size(3,3), sigma);

  recon::StereoSGMParams sgm_params;
  sgm_params.disp_range = 256;
  //sgm_params.window_sz = 1;
  sgm_params.window_sz = 3;
  //sgm_params.penalty1 = 15;
  //sgm_params.penalty1 = 6;         // best - 3 Daimler - 10
  sgm_params.penalty1 = P1;          // best - 3 Daimler - 10
  //sgm_params.penalty2 = 100;
  //sgm_params.penalty2 = 50;        // best - 60 Daimler - 50
  sgm_params.penalty2 = P2;          // best - 60 Daimler - 50
  // Axel tractor
  //sgm_params.disp_range = 270;
  //sgm_params.penalty1 = 7;

  recon::StereoSGM sgm(sgm_params);
  sgm.compute(img_left, img_right, img_disp);
  img_disp.convertTo(img8_disp, CV_8U, 1.0/256.0);

  std::string prefix = left_img_fname;
  size_t slashPosition = prefix.rfind('/');
  if (slashPosition != std::string::npos) prefix.erase(0, slashPosition+1);
  size_t dotPosition = prefix.rfind('.');
  if (dotPosition != std::string::npos) prefix.erase(dotPosition);

  //cv::Mat img8_disp_eq;
  //cv::equalizeHist(img8_disp, img8_disp);
  cv::Mat img_norm_hist;
  cv::imwrite(output_folder + "/disparities/" + prefix + ".png", img_disp);
  cv::normalize(img8_disp, img_norm_hist, 0, 255, cv::NORM_MINMAX);
  cv::imwrite(output_folder + "/norm_hist/" + prefix + ".png", img_norm_hist);
  //cv::imwrite(output_folder + "_disp/" + prefix + "_disp.png", img8_disp);

  //cout << img_disp << "\n\n";
  //imshow("disparity", img_disp);
  //waitKey(0);
}

int main(int argc, char** argv)
{
  if (argc != 6) {
    std::cerr << "usage:\n" << argv[0] << " left right out_folder P1 P2" << std::endl;
    return 1;
  }

  std::string left_img_fname = argv[1];
  std::string right_img_fname = argv[2];
  std::string out_folder = argv[3];
  int P1 = std::stoi(argv[4]);
  int P2 = std::stoi(argv[5]);

  RunSGM(P1, P2, left_img_fname, right_img_fname, out_folder);

  return 0;
}
