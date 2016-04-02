#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <png++/png.hpp>
#include "SGMStereo.h"
#include "SPSStereo.h"
#include "defParameter.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dc1394/dc1394.h>

#include "camera_capture_libdc1394.h"


void ConvertDispImageToCvMat8(const png::image<png::gray_pixel_16>& img, cv::Mat* cvimg) {
  cvimg->create(img.get_height(), img.get_width(), CV_8U);
  for (int i = 0; i < cvimg->rows; i++)
    for (int j = 0; j < cvimg->cols; j++)
      cvimg->at<uint8_t>(i,j) = (uint8_t)std::min(255.0, std::round(img.get_pixel(j,i) / 256.0));
}

void ConvertFloatDispToMat8(const float* disp, const int width, const int height, cv::Mat* img) {
  img->create(height, width, CV_8U);
  for (int i = 0; i < img->rows; i++) {
    for (int j = 0; j < img->cols; j++) {
      float d = disp[i*width + j];
      if (d < 0 || d > 255) throw 1;
      img->at<uint8_t>(i,j) = static_cast<uint8_t>(std::round(d));
    }
  }
}

void ConvertMatToPNG(const cv::Mat& cvimg, png::image<png::rgb_pixel>* img) {
  for (size_t i = 0; i < img->get_height(); i++) {
    for (size_t j = 0; j < img->get_width(); j++) {
      png::rgb_pixel pix;
      pix.red = cvimg.at<uint8_t>(i,j);
      pix.green = cvimg.at<uint8_t>(i,j);
      pix.blue = cvimg.at<uint8_t>(i,j);
      img->set_pixel(j, i, pix);
    }
  }
}

void ConvertRaw12toRaw8(const cv::Mat& raw12, cv::Mat& raw8)
{
  raw8.create(raw12.rows, raw12.cols, CV_8U);
  for (int i = 0; i < raw12.rows; i++) {
    for (int j = 0; j < raw12.cols; j++) {
      raw8.at<uint8_t>(i,j) = uint8_t(255.0 * ((double)raw12.at<uint16_t>(i,j) / 4095.0));
      //std::cout << raw12.at<uint16_t>(i,j) << "\n";
      //printf("%d\n", raw12.at<uint16_t>(i,j));
    }
  }
}

void ConvertRGB12toBGR8(const cv::Mat& rgb12, cv::Mat& bgr8)
{
  bgr8.create(rgb12.rows, rgb12.cols, CV_8UC3);
  for (int i = 0; i < rgb12.rows; i++) {
    for (int j = 0; j < rgb12.cols; j++) {
      bgr8.at<cv::Vec3b>(i,j)[0] = uint8_t(255.0 * ((double)rgb12.at<cv::Vec3s>(i,j)[0] / 4095.0));
      bgr8.at<cv::Vec3b>(i,j)[1] = uint8_t(255.0 * ((double)rgb12.at<cv::Vec3s>(i,j)[1] / 4095.0));
      bgr8.at<cv::Vec3b>(i,j)[2] = uint8_t(255.0 * ((double)rgb12.at<cv::Vec3s>(i,j)[2] / 4095.0));
      //std::cout << raw12.at<uint16_t>(i,j) << "\n";
    }
  }
}


//int main(int argc, char** argv) {
//  if (argc != 2)
//    return -1;
//  std::string calib_path = argv[1];
//  cv::FileStorage fs(calib_path, CV_STORAGE_READ);
//  // calibration params
//  cv::Mat M1, D1, M2, D2, R, T;
//  fs["M1"] >> M1;
//  fs["D1"] >> D1;
//  fs["M2"] >> M2;
//  fs["D2"] >> D2;
//  fs["R"] >> R;
//  fs["T"] >> T;
//
//  cv::Mat R1, R2, P1, P2, Q;
//  //cv::Size new_size = example_img.size();
//  double alpha = 0.0;	   // 0 - will zoom in ROI by default, to disable this set to -1
//  cv::Rect valid_roi[2];
//  cv::Size img_size(640, 480);
//  cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q,
//                    cv::CALIB_ZERO_DISPARITY, alpha, img_size, &valid_roi[0], &valid_roi[1]);
//  cv::Mat left_rmap[2];
//  cv::Mat right_rmap[2];
//  //int mtype = CV_16SC2;
//  int mtype = CV_32FC1;
//  cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, mtype, left_rmap[0], left_rmap[1]);
//  cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, mtype, right_rmap[0], right_rmap[1]);
//  
//  int num_channels = 1;
//  int minDisparity = 0;
//  int numDisparities = 160;;
//  //int numDisparities = 128;
//  int blockSize = 3;
//  //int P_small = 3;
//  //int P_large = 80;
//  int P_small = 8 * num_channels * blockSize * blockSize;
//  int P_large = 32 * num_channels * blockSize * blockSize;
//  //int disp12MaxDiff = 1;
//  int disp12MaxDiff = 1;
//  int preFilterCap = 63;
//  //int preFilterCap = 43;
//  int uniquenessRatio = 10;
//  //int uniquenessRatio = 0;
//  //int speckleWindowSize = 100;
//  int speckleWindowSize = 200;
//  //int speckleRange = 32;
//  int speckleRange = 2;
////int 	mode = StereoSGBM::MODE_SGBM 
////)
//  //cv::StereoSGBM::create(0, 128, 5, 10, 50, 10, 20, 0, 0, 0);
//  cv::StereoSGBM sgm(minDisparity, numDisparities, blockSize, P_small, P_large, disp12MaxDiff,
//                     preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, true);
//
//  dc1394camera_list_t* list;
//  dc1394error_t err;
//  dc1394_t* bus_data = dc1394_new();
//  if (!bus_data) return 1;
//  err = dc1394_camera_enumerate(bus_data, &list);
//  DC1394_ERR_RTN(err, "Failed to enumerate cameras");
//
//  printf("Num of connected cameras = %d\n", list->num);
//  if (list->num != 1) {
//    dc1394_log_error("BB2 camera not found!\n");
//    return 1;
//  }
//
//  bool use_external_trigger = false;
//  uint64_t guid = list->ids[0].guid;
//
//  //for (int i = 0; i < 2; i++) {
//  //  if (list->ids[i].guid == 13602058368603641)
//  //    left_guid = list->ids[i].guid;
//  //  else if (list->ids[i].guid == 13602058368603640)
//  //    right_guid = list->ids[i].guid;
//  //  else throw 1;
//  //}
//  dc1394_camera_free_list(list);
//
//  cam::CameraCaptureLibdc1394 camera(guid, bus_data, use_external_trigger);
//  cv::Mat left_img, right_img;
//  cv::Mat left_rect_img, right_rect_img;
//  cv::Mat left_rgb_img, right_rgb_img;
//  cv::Mat left_disp_img, right_disp_img, left_disp_rawimg;
//  cv::Mat disp_img, disp_img8, display_img;
//
//  std::string save_folder = "/home/kivan/source/cv-stereo/build/bb2_sgm/release/imgs/";
//  //double delta_sum = 0;
//  //double tdiff_prev;
//  uint64_t i = 0;
//  uint64_t save_idx = 0;
//  uint64_t time = 0;
//  auto start = std::chrono::system_clock::now();
//  while (true) {
//  //for (; i < 300; i++) {
//    printf("\nframe = %lu\n", i);
//    time = camera.Grab(&left_img, &right_img);
//    //std::cout << left_img.size() << std::endl;
//    //std::cout << right_img.size() << std::endl;
//
//    cv::remap(left_img, left_rect_img, left_rmap[0], left_rmap[1], CV_INTER_LINEAR);
//    cv::remap(right_img, right_rect_img, right_rmap[0], right_rmap[1], CV_INTER_LINEAR);
//    cv::resize(left_rect_img, left_rect_img, cv::Size(), 0.5, 0.5);
//    cv::resize(right_rect_img, right_rect_img, cv::Size(), 0.5, 0.5);
//    //cv::waitKey(6);
//    sgm(left_rect_img, right_rect_img, disp_img);
//    disp_img.convertTo(disp_img8, CV_8U, 1.0/16.0);
//    cv::normalize(disp_img8, disp_img8, 100, 255, cv::NORM_MINMAX);
//    //cv::equalizeHist(disp_img8, disp_img8);
//    //cv::resize(disp_img8, display_img, 2 * img_size, 2.0, 2.0);
//    cv::resize(disp_img8, display_img, cv::Size(1280, 960));
//
//
//    // get right frame closest to left_time
//    //right_time = right_cam.Grab(right_img);
//    //right_time = right_cam.Grab(right_img, left_time);
//
//    i++;
//
//
//    //cv::imshow("left_image", left_img);
//    //cv::imshow("right_image", right_img);
//    cv::imshow("left_rect_image", left_rect_img);
//    cv::imshow("right_rect_image", right_rect_img);
//    //cv::imshow("depth_img", disp_img8);
//    cv::imshow("depth_img", display_img);
//    //cv::imshow("left_raw_image", left_disp_rawimg);
//    //cv::imshow("right_image", right_disp_img);
//    int key = cv::waitKey(10);
//    if (key == 10) {
//      std::stringstream prefix;
//      prefix << std::setw(6) << std::setfill('0') << save_idx;
//      std::cout << prefix.str() << "\n";
//      cv::imwrite(save_folder + "/left/" + prefix.str() + ".png", left_rect_img);
//      //cv::imwrite(save_folder + "left/" + prefix.str() + "_left_rgb.png", left_disp_img);
//      //cv::imwrite(save_folder + "left/" + prefix.str() + "_left_raw.png", left_disp_rawimg);
//      cv::imwrite(save_folder + "/right/" + prefix.str() + ".png", right_rect_img);
//      cv::imwrite(save_folder + "/disp/" + prefix.str() + ".png", disp_img8);
//      save_idx++;
//    }
//    else if (key == 27) {
//      break;
//    }
//  }
//  auto end = std::chrono::system_clock::now();
//  std::chrono::duration<double> elapsed = end - start;
//  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
//  std::cout << "FPS = " << static_cast<double>(i) / elapsed.count() << "\n";
//  
//  return 0;
//}


//int main(int argc, char** argv) {
//  if (argc != 2)
//    return -1;
//  std::string calib_path = argv[1];
//  cv::FileStorage fs(calib_path, CV_STORAGE_READ);
//  // calibration params
//  cv::Mat M1, D1, M2, D2, R, T;
//  fs["M1"] >> M1;
//  fs["D1"] >> D1;
//  fs["M2"] >> M2;
//  fs["D2"] >> D2;
//  fs["R"] >> R;
//  fs["T"] >> T;
//
//  cv::Mat R1, R2, P1, P2, Q;
//  //cv::Size new_size = example_img.size();
//  double alpha = 0.0;	   // 0 - will zoom in ROI by default, to disable this set to -1
//  cv::Rect valid_roi[2];
//  cv::Size img_size(640, 480);
//  cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q,
//                    cv::CALIB_ZERO_DISPARITY, alpha, img_size, &valid_roi[0], &valid_roi[1]);
//  cv::Mat left_rmap[2];
//  cv::Mat right_rmap[2];
//  //int mtype = CV_16SC2;
//  int mtype = CV_32FC1;
//  cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, mtype, left_rmap[0], left_rmap[1]);
//  cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, mtype, right_rmap[0], right_rmap[1]);
//
//  png::image<png::rgb_pixel> left_png(img_size.width, img_size.height);
//  png::image<png::rgb_pixel> right_png(img_size.width, img_size.height);
//  SPSStereo sps;
//  sps.setIterationTotal(outerIterationTotal, innerIterationTotal);
//  sps.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
//  sps.setInlierThreshold(lambda_d);
//  sps.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);
//
//  png::image<png::gray_pixel_16> segmentImage;
//  png::image<png::gray_pixel_16> disparityImage;
//  std::vector<std::vector<double>> disparityPlaneParameters;
//  std::vector<std::vector<int>> boundaryLabels;
//
//  //float* disparities = reinterpret_cast<float*>(malloc(width * height * sizeof(float)));
//  //float* disparities = new float[img_size.width * img_size.height];
//  
//  dc1394camera_list_t* list;
//  dc1394error_t err;
//  dc1394_t* bus_data = dc1394_new();
//  if (!bus_data) return 1;
//  err = dc1394_camera_enumerate(bus_data, &list);
//  DC1394_ERR_RTN(err, "Failed to enumerate cameras");
//
//  printf("Num of connected cameras = %d\n", list->num);
//  if (list->num != 1) {
//    dc1394_log_error("BB2 camera not found!\n");
//    return 1;
//  }
//
//  bool use_external_trigger = false;
//  uint64_t guid = list->ids[0].guid;
//
//  //for (int i = 0; i < 2; i++) {
//  //  if (list->ids[i].guid == 13602058368603641)
//  //    left_guid = list->ids[i].guid;
//  //  else if (list->ids[i].guid == 13602058368603640)
//  //    right_guid = list->ids[i].guid;
//  //  else throw 1;
//  //}
//  dc1394_camera_free_list(list);
//
//  cam::CameraCaptureLibdc1394 camera(guid, bus_data, use_external_trigger);
//  cv::Mat left_img, right_img;
//  cv::Mat left_rect_img, right_rect_img;
//  cv::Mat left_rgb_img, right_rgb_img;
//  cv::Mat left_disp_img, right_disp_img, left_disp_rawimg;
//  cv::Mat disp_img, disp_img8;
//
//  std::string save_folder = "/home/kivan/source/cv-stereo/build/bb2_sgm/release/imgs/";
//  //double delta_sum = 0;
//  //double tdiff_prev;
//  uint64_t i = 0;
//  uint64_t save_idx = 0;
//  uint64_t time = 0;
//  auto start = std::chrono::system_clock::now();
//  while (true) {
//  //for (; i < 300; i++) {
//    printf("\nframe = %lu\n", i);
//    time = camera.Grab(&left_img, &right_img);
//    //std::cout << left_img.size() << std::endl;
//    //std::cout << right_img.size() << std::endl;
//
//    cv::remap(left_img, left_rect_img, left_rmap[0], left_rmap[1], CV_INTER_LINEAR);
//    cv::remap(right_img, right_rect_img, right_rmap[0], right_rmap[1], CV_INTER_LINEAR);
//    //cv::waitKey(6);
//    ConvertMatToPNG(left_rect_img, &left_png);
//    ConvertMatToPNG(right_rect_img, &right_png);
//
//    sps.compute(superpixelTotal, left_png, right_png, segmentImage, disparityImage,
//                disparityPlaneParameters, boundaryLabels);
//    ConvertDispImageToCvMat8(disparityImage, &disp_img8);
//
//    //sgm(left_rect_img, right_rect_img, disp_img);
//    //disp_img.convertTo(disp_img8, CV_8U, 1.0/16.0);
//    //cv::normalize(disp_img8, disp_img8, 0, 255, cv::NORM_MINMAX);
//    cv::equalizeHist(disp_img8, disp_img8);
//
//
//    // get right frame closest to left_time
//    //right_time = right_cam.Grab(right_img);
//    //right_time = right_cam.Grab(right_img, left_time);
//
//    printf("left timestamp = %lu us\n", time);
//    //printf("right timestamp = %lu us\n", right_time);
//    //double tdiff = static_cast<int>(right_time - left_time) / 1000.0;
//    //printf("time diff = %.2f ms\n", tdiff);
//    //if (i > 0)
//    //  delta_sum += (tdiff - tdiff_prev);
//    i++;
//    //tdiff_prev = tdiff;
//    //printf("delta time = %f msec\n", dt_msec);
//    //printf("delta sum = %.2f ms\n\n", delta_sum);
//
//    //if (left_img.channels() == 2) {
//    //  cv::cvtColor(left_img, left_disp_img, CV_YUV2BGR_UYVY);
//    //  cv::cvtColor(right_img, right_disp_img, CV_YUV2BGR_UYVY);
//    //}
//    //else {
//    //  //left_img.copyTo(left_disp_img);
//    //  //right_img.copyTo(right_disp_img);
//    //  ConvertRaw12toRaw8(left_img, left_disp_rawimg);
//    //  ConvertRaw12toRaw8(right_img, right_disp_img);
//
//    //  // DC1394_BAYER_METHOD_NEAREST=0,
//    //  // DC1394_BAYER_METHOD_SIMPLE,
//    //  // DC1394_BAYER_METHOD_BILINEAR,
//    //  // DC1394_BAYER_METHOD_HQLINEAR,
//    //  // DC1394_BAYER_METHOD_DOWNSAMPLE,
//    //  // DC1394_BAYER_METHOD_EDGESENSE,
//    //  // DC1394_BAYER_METHOD_VNG,
//    //  // DC1394_BAYER_METHOD_AHD
//
//    //  left_rgb_img.create(left_img.rows, left_img.cols, CV_16UC3);
//    //  dc1394_bayer_decoding_16bit((uint16_t*)left_img.data, (uint16_t*)left_rgb_img.data,
//    //                              left_img.cols, left_img.rows, DC1394_COLOR_FILTER_RGGB,
//    //                              DC1394_BAYER_METHOD_BILINEAR, 12);
//    //  ConvertRGB12toBGR8(left_rgb_img, left_disp_img);
//    //  right_rgb_img.create(right_img.rows, right_img.cols, CV_16UC3);
//    //  dc1394_bayer_decoding_16bit((uint16_t*)right_img.data, (uint16_t*)right_rgb_img.data,
//    //                              right_img.cols, right_img.rows, DC1394_COLOR_FILTER_RGGB,
//    //                              DC1394_BAYER_METHOD_BILINEAR, 12);
//    //  ConvertRGB12toBGR8(right_rgb_img, right_disp_img);
//    //}
//
//    //cv::imshow("left_image", left_img);
//    //cv::imshow("right_image", right_img);
//    cv::imshow("left_rect_image", left_rect_img);
//    cv::imshow("right_rect_image", right_rect_img);
//    cv::imshow("disp_img", disp_img8);
//    //cv::imshow("left_raw_image", left_disp_rawimg);
//    //cv::imshow("right_image", right_disp_img);
//    int key = cv::waitKey(10);
//    if (key == 10) {
//      std::stringstream prefix;
//      prefix << std::setw(6) << std::setfill('0') << save_idx;
//      std::cout << prefix.str() << "\n";
//      cv::imwrite(save_folder + "/left/" + prefix.str() + ".png", left_rect_img);
//      //cv::imwrite(save_folder + "left/" + prefix.str() + "_left_rgb.png", left_disp_img);
//      //cv::imwrite(save_folder + "left/" + prefix.str() + "_left_raw.png", left_disp_rawimg);
//      cv::imwrite(save_folder + "/right/" + prefix.str() + ".png", right_rect_img);
//      cv::imwrite(save_folder + "/disp/" + prefix.str() + ".png", disp_img8);
//      save_idx++;
//    }
//    else if (key == 27) {
//      break;
//    }
//  }
//  auto end = std::chrono::system_clock::now();
//  std::chrono::duration<double> elapsed = end - start;
//  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
//  std::cout << "FPS = " << static_cast<double>(i) / elapsed.count() << "\n";
//  
//  return 0;
//}


int main(int argc, char** argv) {
  if (argc != 2)
    return -1;
  std::string calib_path = argv[1];
  cv::FileStorage fs(calib_path, CV_STORAGE_READ);
  // calibration params
  cv::Mat M1, D1, M2, D2, R, T;
  fs["M1"] >> M1;
  fs["D1"] >> D1;
  fs["M2"] >> M2;
  fs["D2"] >> D2;
  fs["R"] >> R;
  fs["T"] >> T;

  cv::Mat R1, R2, P1, P2, Q;
  //cv::Size new_size = example_img.size();
  double alpha = 0.0;	   // 0 - will zoom in ROI by default, to disable this set to -1
  cv::Rect valid_roi[2];
  cv::Size img_size(640, 480);
  cv::Size img_size_small(640, 480);
  cv::stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, alpha, img_size, &valid_roi[0], &valid_roi[1]);
  cv::Mat left_rmap[2];
  cv::Mat right_rmap[2];
  //int mtype = CV_16SC2;
  int mtype = CV_32FC1;
  cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, mtype, left_rmap[0], left_rmap[1]);
  cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, mtype, right_rmap[0], right_rmap[1]);

  png::image<png::rgb_pixel> left_png(img_size_small.width, img_size_small.height);
  png::image<png::rgb_pixel> right_png(img_size_small.width, img_size_small.height);
  SGMStereo sgm;
  sgm.setDisparityTotal(128);

  //float* disparities = reinterpret_cast<float*>(malloc(width * height * sizeof(float)));
  float* disparities = new float[img_size_small.width * img_size_small.height];
  
  dc1394camera_list_t* list;
  dc1394error_t err;
  dc1394_t* bus_data = dc1394_new();
  if (!bus_data) return 1;
  err = dc1394_camera_enumerate(bus_data, &list);
  DC1394_ERR_RTN(err, "Failed to enumerate cameras");

  printf("Num of connected cameras = %d\n", list->num);
  if (list->num != 1) {
    dc1394_log_error("BB2 camera not found!\n");
    return 1;
  }

  bool use_external_trigger = false;
  uint64_t guid = list->ids[0].guid;

  //for (int i = 0; i < 2; i++) {
  //  if (list->ids[i].guid == 13602058368603641)
  //    left_guid = list->ids[i].guid;
  //  else if (list->ids[i].guid == 13602058368603640)
  //    right_guid = list->ids[i].guid;
  //  else throw 1;
  //}
  dc1394_camera_free_list(list);

  cam::CameraCaptureLibdc1394 camera(guid, bus_data, use_external_trigger);
  cv::Mat left_img, right_img;
  cv::Mat left_rect_img, right_rect_img;
  cv::Mat left_rgb_img, right_rgb_img;
  cv::Mat left_disp_img, right_disp_img, left_disp_rawimg;
  cv::Mat disp_img, disp_img8;

  std::string save_folder = "/home/kivan/source/cv-stereo/build/bb2_sgm/release/imgs/";
  //double delta_sum = 0;
  //double tdiff_prev;
  //uint64_t i = 0;
  uint64_t save_idx = 0;
  uint64_t time = 0;
  uint64_t num_images = 1000;
  auto start = std::chrono::system_clock::now();
  //while (true) {
  for (int i = 0; i < num_images; i++) {
    //printf("\nframe = %lu\n", i);
    time = camera.Grab(&left_img, &right_img);
    //std::cout << left_img.size() << std::endl;
    //std::cout << right_img.size() << std::endl;

    cv::remap(left_img, left_rect_img, left_rmap[0], left_rmap[1], CV_INTER_LINEAR);
    cv::remap(right_img, right_rect_img, right_rmap[0], right_rmap[1], CV_INTER_LINEAR);

    //cv::resize(left_rect_img, left_rect_img, img_size_small);
    //cv::resize(right_rect_img, right_rect_img, img_size_small);
    ////cv::waitKey(6);
    //ConvertMatToPNG(left_rect_img, &left_png);
    //ConvertMatToPNG(right_rect_img, &right_png);
    //sgm.compute(left_png, right_png, disparities);
    //ConvertFloatDispToMat8(disparities, left_rect_img.cols, left_rect_img.rows, &disp_img8);

    ////sgm(left_rect_img, right_rect_img, disp_img);
    ////disp_img.convertTo(disp_img8, CV_8U, 1.0/16.0);
    //cv::normalize(disp_img8, disp_img8, 0, 255, cv::NORM_MINMAX);
    ////cv::equalizeHist(disp_img8, disp_img8);
    //cv::resize(disp_img8, disp_img8, cv::Size(1280, 960));


    // get right frame closest to left_time
    //right_time = right_cam.Grab(right_img);
    //right_time = right_cam.Grab(right_img, left_time);

    //printf("left timestamp = %lu us\n", time);
    //printf("right timestamp = %lu us\n", right_time);
    //double tdiff = static_cast<int>(right_time - left_time) / 1000.0;
    //printf("time diff = %.2f ms\n", tdiff);
    //if (i > 0)
    //  delta_sum += (tdiff - tdiff_prev);
    //tdiff_prev = tdiff;
    //printf("delta time = %f msec\n", dt_msec);
    //printf("delta sum = %.2f ms\n\n", delta_sum);

    //if (left_img.channels() == 2) {
    //  cv::cvtColor(left_img, left_disp_img, CV_YUV2BGR_UYVY);
    //  cv::cvtColor(right_img, right_disp_img, CV_YUV2BGR_UYVY);
    //}
    //else {
    //  //left_img.copyTo(left_disp_img);
    //  //right_img.copyTo(right_disp_img);
    //  ConvertRaw12toRaw8(left_img, left_disp_rawimg);
    //  ConvertRaw12toRaw8(right_img, right_disp_img);

    //  // DC1394_BAYER_METHOD_NEAREST=0,
    //  // DC1394_BAYER_METHOD_SIMPLE,
    //  // DC1394_BAYER_METHOD_BILINEAR,
    //  // DC1394_BAYER_METHOD_HQLINEAR,
    //  // DC1394_BAYER_METHOD_DOWNSAMPLE,
    //  // DC1394_BAYER_METHOD_EDGESENSE,
    //  // DC1394_BAYER_METHOD_VNG,
    //  // DC1394_BAYER_METHOD_AHD

    //  left_rgb_img.create(left_img.rows, left_img.cols, CV_16UC3);
    //  dc1394_bayer_decoding_16bit((uint16_t*)left_img.data, (uint16_t*)left_rgb_img.data,
    //                              left_img.cols, left_img.rows, DC1394_COLOR_FILTER_RGGB,
    //                              DC1394_BAYER_METHOD_BILINEAR, 12);
    //  ConvertRGB12toBGR8(left_rgb_img, left_disp_img);
    //  right_rgb_img.create(right_img.rows, right_img.cols, CV_16UC3);
    //  dc1394_bayer_decoding_16bit((uint16_t*)right_img.data, (uint16_t*)right_rgb_img.data,
    //                              right_img.cols, right_img.rows, DC1394_COLOR_FILTER_RGGB,
    //                              DC1394_BAYER_METHOD_BILINEAR, 12);
    //  ConvertRGB12toBGR8(right_rgb_img, right_disp_img);
    //}

    ////cv::imshow("left_image", left_img);
    ////cv::imshow("right_image", right_img);
    //cv::imshow("left_rect_image", left_rect_img);
    //cv::imshow("right_rect_image", right_rect_img);
    //cv::imshow("disp_img", disp_img8);
    ////cv::imshow("left_raw_image", left_disp_rawimg);
    ////cv::imshow("right_image", right_disp_img);
    //int key = cv::waitKey(10);
    //if (key == 10) {
    //  std::stringstream prefix;
    //  prefix << std::setw(6) << std::setfill('0') << save_idx;
    //  std::cout << prefix.str() << "\n";
    //  cv::imwrite(save_folder + "/left/" + prefix.str() + ".png", left_rect_img);
    //  //cv::imwrite(save_folder + "left/" + prefix.str() + "_left_rgb.png", left_disp_img);
    //  //cv::imwrite(save_folder + "left/" + prefix.str() + "_left_raw.png", left_disp_rawimg);
    //  cv::imwrite(save_folder + "/right/" + prefix.str() + ".png", right_rect_img);
    //  cv::imwrite(save_folder + "/disp/" + prefix.str() + ".png", disp_img8);
    //  save_idx++;
    //}
    //else if (key == 27) {
    //  break;
    //}
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
  std::cout << "FPS = " << static_cast<double>(num_images) / elapsed.count() << "\n";
  
  return 0;
}
