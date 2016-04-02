#include <iostream>
#include <fstream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv) {
  std::ofstream calib_file("calib_igor.txt");
  std::string left_folder = "/home/kivan/datasets/Egomotion/igor/raw/left/";
  std::string right_folder = "/home/kivan/datasets/Egomotion/igor/raw/right/";
  std::string left_output_folder = "/home/kivan/datasets/Egomotion/igor/rectified/left/";
  std::string right_output_folder = "/home/kivan/datasets/Egomotion/igor/rectified/right/";
  cv::Mat example_img = cv::imread("/home/kivan/datasets/Egomotion/igor/raw/left/00001.png", CV_LOAD_IMAGE_ANYDEPTH);
  std::cout << "Bytes per pixel: " << example_img.step / example_img.cols << "\n";

  // LENSES 3.15 mm - druga KALIBRACIJA...
  cv::Mat K1 = (cv::Mat_<double>(3,3) << 527.1942890450666,0.0,392.76292799400403,
                           0.0,	528.318239665181, 244.53269934266552,
                           0.0,	0.0, 1.0);

  cv::Mat D1 = (cv::Mat_<double>(5,1) << -0.39138285262160577, 0.1419481600238573, 0.0018628385478359263, 0.0011819898895649445, 0.0);

  cv::Mat K2 = (cv::Mat_<double>(3,3) << 524.5384699571192,	0.0,	358.9670102317409,
      0.0,	524.8642070332819,	229.5667921040571,
      0.0,	0.0,	1.0);

  cv::Mat D2 = (cv::Mat_<double>(5,1) << -0.40247524732676826, 0.15349197605160922, -0.0016729952127136878,-0.00106020074385726, 0.0);

  cv::Mat R = (cv::Mat_<double>(3,3) << 0.9999618847587798, -0.0035738859318102814, -0.007965950603418977,
                                        0.0035003421702339615, 0.9999513023325947, -0.009227164680445368,
                                        0.007998539514247862, 0.009198929432014643, 0.999925698771136 );

  cv::Mat T = (cv::Mat_<double>(3,1) << -0.1098148193505109, -0.0009166722661622356, 0.00018746300764305892);
  cv::Mat R1, R2, P1, P2, Q;

  cv::Size new_size = example_img.size();
  double alpha = 0.0;	   // 0 - will zoom in ROI by default, to disable this set to -1
  cv::Rect valid_roi[2];
  cv::stereoRectify(K1, D1, K2, D2, example_img.size(), R, T, R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, alpha, new_size, &valid_roi[0], &valid_roi[1]);
  //cv::Mat new_K1 = P1(cv::Rect(0,0,3,3)).clone();
  //cv::Mat new_K2 = P2(cv::Rect(0,0,3,3)).clone();
  std::cout << P1 << "\n" << P2 << "\n";
  //std::cout << new_size << "\n";
  calib_file << std::setprecision(10) << P2.at<double>(0,0) << " " << P2.at<double>(1,1) << " "
             << P2.at<double>(0,2) << " " << P2.at<double>(1,2) << " "
             << std::abs(P2.at<double>(0,3) / P2.at<double>(0,0)) << "\n";

  cv::Mat left_rmap[2];
  cv::Mat right_rmap[2];
  //int mtype = CV_16SC2;
  int mtype = CV_32FC1;
  cv::initUndistortRectifyMap(K1, D1, R1, P1, new_size, mtype, left_rmap[0], left_rmap[1]);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, new_size, mtype, right_rmap[0], right_rmap[1]);

  //visual_odometer_params_.base = fabs(P2.at<double>(0,3)/P2.at<double>(0,0));
  //visual_odometer_params_.calib.f = newCameraMatrix1.at<double>(0,0);
  //visual_odometer_params_.calib.cu = newCameraMatrix1.at<double>(0,2);
  //visual_odometer_params_.calib.cv = newCameraMatrix1.at<double>(1,2);

  // prije svake slike
  //cv::remap( iLraw, iL, map1left, map2left, CV_INTER_LINEAR);
  //cv::remap( iRraw, iR, map1right, map2right, CV_INTER_LINEAR);
  cv::Mat img_left, img_right, left_rectif, right_rectif;
  for (int i = 1; i <= 3521; i++) {
    std::stringstream filename;
    filename << std::setw(5) << std::setfill('0') << i << ".png";
    img_left = cv::imread(left_folder + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(right_folder + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
    cv::remap(img_left, left_rectif, left_rmap[0], left_rmap[1], CV_INTER_LINEAR);
    cv::remap(img_right, right_rectif, right_rmap[0], right_rmap[1], CV_INTER_LINEAR);
    cv::imwrite(left_output_folder + filename.str(), left_rectif);
    cv::imwrite(right_output_folder + filename.str(), right_rectif);
  }

  return 0;
}

