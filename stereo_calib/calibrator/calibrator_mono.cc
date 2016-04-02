// usage:
//./calibrator_mono witdh height square_size_in_meters -o calib.yml /path/to/imgs/
//./calibrator_mono -w 8 -h 6 -s 0.0372 -o bb_right.yml calib_monitor_right.xml

#include <iostream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "calib_helper.h"

bool RunCalibration(const std::vector<std::vector<cv::Point2f>>& image_points,
                    cv::Size image_size, cv::Size board_size, double square_size,
                    cv::Mat& K, cv::Mat& dist_coeffs, std::vector<cv::Mat>& rvecs,
                    std::vector<cv::Mat>& tvecs, std::vector<double>& reproj_errors,
                    double& reproj_error) {
  K = cv::Mat::eye(3, 3, CV_64F);
  dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

  std::vector<std::vector<cv::Point3f>> object_points(1);
  for(int i = 0; i < board_size.height; i++)
    for(int j = 0; j < board_size.width; j++)
      object_points[0].push_back(cv::Point3f(float(j*square_size), float(i*square_size), 0));
  object_points.resize(image_points.size(), object_points[0]);

  reproj_error = cv::calibrateCamera(object_points, image_points, image_size, K, dist_coeffs,
                                     rvecs, tvecs, cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);
                                            //rvecs, tvecs, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
  ///*|CV_CALIB_FIX_K3*/|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
  printf("RMS error reported by calibrateCamera: %g\n", reproj_error);

  bool ok = cv::checkRange(K) && cv::checkRange(dist_coeffs);

  //totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
  //    rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

  return ok;
}

void FindChessboards(const std::vector<std::string>& img_files, const cv::Size& board_size,
    cv::Size& image_size, std::vector<std::vector<cv::Point2f>>& image_points) {
  bool show_results = true;
  for (size_t i = 0; i < img_files.size(); i++) {
    std::cout << "Processing image: " << img_files[i] << "... ";
    cv::Mat img = cv::imread(img_files[i], cv::IMREAD_COLOR);
    if (i == 0)
      image_size = img.size();
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> keypoints;

    bool found = cv::findChessboardCorners(img, board_size, keypoints,
                    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!found)
      found = cv::findChessboardCorners(img, board_size, keypoints, cv::CALIB_CB_ADAPTIVE_THRESH);
    if (!found)
      found = cv::findChessboardCorners(img, board_size, keypoints, cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
      std::cout << "FAILED!\n";
      continue;
    }
    std::cout << "SUCCESS!\n";
    cv::cornerSubPix(img_gray, keypoints, cv::Size(11,11), cv::Size(-1,-1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    image_points.push_back(keypoints);
    if (show_results) {
      cv::drawChessboardCorners(img, board_size, cv::Mat(keypoints), found);
      cv::imshow("Image", img);
      cv::waitKey(0);
    }
  }
}

void SaveCalibration(const std::string& filename, cv::Size image_size, cv::Size board_size,
                     double square_size, const cv::Mat& K, const cv::Mat& dist_coeffs,
                     double reproj_error)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  //time_t tt;
  //time(&tt);
  //struct tm *t2 = localtime( &tt );
  //char buf[1024];
  //strftime( buf, sizeof(buf)-1, "%c", t2 );
  //fs << "calibration_time" << buf;

  fs << "image_width" << image_size.width;
  fs << "image_height" << image_size.height;
  fs << "board_width" << board_size.width;
  fs << "board_height" << board_size.height;
  fs << "square_size" << square_size;

  fs << "camera_matrix" << K;
  fs << "distortion_coefficients" << dist_coeffs;

  fs << "reprojection_error" << reproj_error;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cout << argv[0] << " witdh height square_size_in_meters /path/to/imgs/ calib.yml\n";
    return 1;
  }
  cv::Size board_size, image_size;
  board_size.width = std::stoi(argv[1]);
  board_size.height = std::stoi(argv[2]);
  double square_size = std::stod(argv[3]);
  std::string img_folder = argv[4];
  std::string calib_filename = argv[5];

  std::vector<std::vector<cv::Point2f>> image_points;
  std::vector<std::string> img_files;
  CalibHelper::GetFilesInFolder(img_folder, img_files);
  FindChessboards(img_files, board_size, image_size, image_points);

  cv::Mat K, dist_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  std::vector<double> reproj_errors;
  double reproj_error;
  bool ok = RunCalibration(image_points, image_size, board_size, square_size, K, dist_coeffs,
                           rvecs, tvecs, reproj_errors, reproj_error);
  if (!ok) {
    std::cout << "Calibration failed!\n";
  }
  std::cout << "Calibration done!\n";
  std::cout << "Camera matrix:\n" << K << "\n\nDistortion coefficients:\n" << dist_coeffs << "\n";
  SaveCalibration(calib_filename, image_size, board_size, square_size, K, dist_coeffs,
                  reproj_error);

  return 0;
}
