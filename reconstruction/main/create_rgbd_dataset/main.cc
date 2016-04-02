#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../base/stereo_sgm.h"
#include "../../../core/format_helper.h"

void CalculateDepthImage(const cv::Mat& disp_img, const double* calib, cv::Mat& depth_img) {
  double f = calib[0];
  double b = calib[4];
  depth_img = cv::Mat::zeros(disp_img.rows, disp_img.cols, CV_16U);
  for (int i = 0; i < disp_img.rows; i++) {
    for (int j = 0; j < disp_img.cols; j++) {
      double d = disp_img.at<double>(i,j);
      if (d >= 1.0) {
        // in cm
        double z = 100.0 * f * b / d;
        if (z >= 65536) throw 1;
        depth_img.at<uint16_t>(i,j) = static_cast<uint16_t>(std::round(z));
      }
    }
  }
}

void FixKittiGroundtruth(cv::Mat& img_gt) {
  for (int i = 0; i < img_gt.rows; i++) {
    for (int j = 0; j < img_gt.cols; j++) {
      uint8_t& b = img_gt.at<cv::Vec3b>(i,j)[0];
      uint8_t& g = img_gt.at<cv::Vec3b>(i,j)[1];
      uint8_t& r = img_gt.at<cv::Vec3b>(i,j)[2];
      if (r == 255 && g == 0 && b == 0) {
        r = 0;
        g = 0;
        b = 255;
      }
      else {
        r = 0;
        g = 255;
        b = 0;
      }
    }
  }
}

void RunSGM(const int P1, const int P2, const std::string data_path, const std::string filename,
            const std::string output_folder, const double* calib, bool have_groundtruth) {
  //cv::Mat img_disp, img_disp_subpixel, mat_disp;
  cv::Mat img_disp, img8_disp;
  cv::Mat img_left = cv::imread(data_path + "/left/" + filename, cv::IMREAD_GRAYSCALE);
  cv::Mat img_right = cv::imread(data_path + "/right/" + filename, cv::IMREAD_GRAYSCALE);
  cv::Mat img_left_color = cv::imread(data_path + "/left/" + filename, cv::IMREAD_COLOR);
  
  cv::Mat img_groundtruth;
  if (have_groundtruth) {
    size_t pos = filename.find('_');
    std::string gt_filename = filename.substr(0, pos) + "_road_" + filename.substr(pos+1);
    img_groundtruth = cv::imread(data_path + "/groundtruth/" + gt_filename, cv::IMREAD_COLOR);
    FixKittiGroundtruth(img_groundtruth);
  }
  //cv::imshow("img_left", img_left);
  //cv::waitKey(0);

  //double sigma = 0.7;
  //cv::GaussianBlur(img_left, img_left, cv::Size(3,3), sigma);
  //cv::GaussianBlur(img_right, img_right, cv::Size(3,3), sigma);

  recon::StereoSGMParams sgm_params;
  //sgm_params.disp_range = 256;
  sgm_params.disp_range = 120;
  //sgm_params.window_sz = 1;
  sgm_params.window_sz = 5;
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

  cv::Mat img_disp_float;
  img_disp.convertTo(img_disp_float, CV_64F, 1.0/256.0);
  cv::Mat depth_image;
  CalculateDepthImage(img_disp_float, calib, depth_image);

  //std::string prefix = left_img_fname;
  //size_t slashPosition = prefix.rfind('/');
  //if (slashPosition != std::string::npos) prefix.erase(0, slashPosition+1);
  //size_t dotPosition = prefix.rfind('.');
  //if (dotPosition != std::string::npos) prefix.erase(dotPosition);

  cv::equalizeHist(img8_disp, img8_disp);
  std::string prefix = filename.substr(0, filename.size() - 4);
  cv::imwrite(output_folder + "/" + prefix + "_colors.png", img_left_color);
  cv::imwrite(output_folder + "/" + prefix + "_depth.png", depth_image);
  cv::imwrite(output_folder + "/" + prefix + "_depth_visualization.png", img8_disp);
  if (have_groundtruth)
    cv::imwrite(output_folder + "/" + prefix + "_ground_truth.png", img_groundtruth);
  else {
    img_groundtruth = cv::Mat::zeros(img_left.rows, img_left.cols, CV_8UC3);
    cv::imwrite(output_folder + "/" + prefix + "_ground_truth.png", img_groundtruth);
  }
  //cv::imwrite(output_folder + "_disp/" + prefix + "_disp.png", img8_disp);

  //cout << img_disp << "\n\n";
  //imshow("disparity", img_disp);
  //waitKey(0);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage:\n" << argv[0] << " -train/-test data_folder out_folder" << std::endl;
    return 1;
  }

  std::string data_type = argv[1];
  std::string data_path = argv[2];
  std::string out_folder = argv[3];
  bool have_groundtruth;
  if (data_type == "-train")
    have_groundtruth = true;
  else if (data_type == "-test")
    have_groundtruth = false;
  else
    return 1;

  int P1 = 5;
  int P2 = 60;

  double mono_cam[5];
  double color_cam[5];

  std::vector<std::string> files;
  core::FormatHelper::GetFilesInFolder(data_path + "/left/", files, false);

  for (const auto& file : files) {
    std::cout << file << "\n";
    std::string calib_path = data_path + "/calib/" + file.substr(0, file.size()-4) + ".txt";
    core::FormatHelper::ReadCalibKitti(calib_path, mono_cam, color_cam);
    RunSGM(P1, P2, data_path, file, out_folder, color_cam, have_groundtruth);
  }

  return 0;
}
