#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../../base/stereo_cvsgbm.h"
#include "../../base/stereo_sgm.h"

int main(int argc, char** argv)
{
  //std::vector<int> P1 = { 20 };
  //std::vector<int> P2 = { 800 };
  // ZSAD
  //std::vector<int> P1 = {1, 5, 10, 20, 30, 40 };
  //std::vector<int> P2 = {50, 60, 100, 200, 400, 600, 800, 1000, 1400, 1800 };

  // Census 4*P1 + 4*P2 < 255
  std::vector<int> P1 = { 1, 3, 5, 7, 10, 15, 20 };
  std::vector<int> P2 = { 20, 30, 35, 40, 50, 60, 70, 80, 100, 150, 250 };

  recon::StereoSGMParams sgm_params;
  sgm_params.disp_range = 128;
  //sgm_params.window_sz = 1;
  sgm_params.window_sz = 5;
  //cv::Mat img_left = cv::imread("/home/kivan/Projects/datasets/Middlebury/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
  //cv::Mat img_right = cv::imread("/home/kivan/Projects/datasets/Middlebury/im6.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_left = cv::imread("/home/kivan/Projects/datasets/KITTI/sequences_gray/01/image_0/000102.png",
                                CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread("/home/kivan/Projects/datasets/KITTI/sequences_gray/01/image_1/000102.png",
                                 CV_LOAD_IMAGE_GRAYSCALE);
  double sigma = 0.7;
  cv::GaussianBlur(img_left, img_left, cv::Size(3,3), sigma);
  cv::GaussianBlur(img_right, img_right, cv::Size(3,3), sigma);

  for(size_t i = 0; i < P1.size(); i++) {
    for(size_t j = 0; j < P2.size(); j++) {
      sgm_params.penalty1 = P1[i];
      sgm_params.penalty2 = P2[j];
      recon::StereoSGM sgm(sgm_params);
      cv::Mat img_disp, mat_disp;
      struct timeval start, end;
      gettimeofday(&start, NULL);
      sgm.compute(img_left, img_right, mat_disp);
      gettimeofday(&end, NULL);
      float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
      //std::cout << "\nMy SGM time spent = " << delta << "s\n\n";
      printf("\nMy SGM time spent = %fs\n\n", delta);
      mat_disp.convertTo(img_disp, CV_8U);
      cv::equalizeHist(img_disp, img_disp);
      std::stringstream P_str;
      P_str << "_P1_" << std::setw(5) << std::setfill('0') << P1[i] << "_P2_" << std::setw(5) << std::setfill('0') << P2[j];
      cv::imwrite("disp" + P_str.str() + ".png", img_disp);
    }
  }

  //recon::StereoSGBMParams params(min_disparity, number_of_disparities, sad_window_size, P1, P2, disp12_max_diff,
  //                               pre_filter_cap, uniqueness_ratio, speckle_window_size, speckle_range, full_dp);
  //cv::Mat buffer;
  //recon::computeDisparitySGBM(img_left, img_right, img_disp, params, buffer);
  //recon::computeSGM(img_left, img_right, params, img_disp);
  //sgbm(img_left, img_right, img_disp);

  //img_disp.convertTo(img_disp_save, CV_8U, 1.0/16.0);
  //img_disp.convertTo(mat_disp_save, CV_32F, 1.0/16.0);
  //for(int j = 0; j < img_disp.rows; j++) {
  //  for(int k = 0; k < img_disp.cols; k++) {
  //    std::cout << (int)img_disp_save.at<uint8_t>(j,k) << " - CV_8U\n";
  //    std::cout << img_disp.at<int16_t>(j,k) / 16.0 << " - CV_16S\n";
  //    std::cout << mat_disp_save.at<float>(j,k) << " - CV_32F\n";
  //  }
  //}
  //img_disp.convertTo(img_disp, CV_8U, (255.0/sgbm.numberOfDisparities) * (1.0/16.0));
  //cv::equalizeHist(img_disp, img_disp);
  //cv::imshow("Disparity image", img_disp);

  //string img_fname = output_folder + "img/" + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.png";
  //string mat_fname = output_folder + "mat/" + imagelist[i].substr(9, imagelist[i].size()-13) + "_disp.yml";
  //cout << "Saving: " << img_fname << "\n" << mat_fname << "\n----------------------\n";
  //cv::imwrite(img_fname, img_disp_save, compression_params);
  //cv::FileStorage mat_file(mat_fname, cv::FileStorage::WRITE);
  //mat_file << "disparity_matrix" << mat_disp_save;

  //cout << img_disp << "\n\n";
  //imshow("disparity", img_disp);
  //waitKey(0);
}
