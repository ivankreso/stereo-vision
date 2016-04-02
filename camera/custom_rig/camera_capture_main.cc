#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdio>

#include <opencv2/highgui/highgui.hpp>
#include <dc1394/dc1394.h>

#include <opencv2/imgproc/imgproc.hpp>

#include "camera_capture_libdc1394.h"

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

void SynchronizeCameras(cam::CameraCaptureLibdc1394& left_cam,
                        cam::CameraCaptureLibdc1394& right_cam,
                        cv::Mat& left_img, cv::Mat& right_img)
{
  int fps = 10;
  //double max_tdiff = 1000.0 / fps / 4.0;    // hardcoded - 2ms
  double max_tdiff = 20.0;
  uint64_t left_time, right_time;
  left_time = left_cam.Grab(left_img);
  right_time = right_cam.Grab(right_img);

  while (true) {
    double tdiff = static_cast<int>(right_time - left_time) / 1000.0;
    printf("time diff = %.2f ms\n", tdiff);
    // if the images are unsynced - sync them
    if (std::abs(tdiff) > max_tdiff) {
      // skip left image
      if (left_time < right_time) {
        left_time = left_cam.Grab(left_img);
      }
      // skip right image
      else {
        right_time = right_cam.Grab(right_img);
      }
    }
    else break;
  }
}

int main(int argc, char** argv) {
  dc1394camera_list_t* list;
  dc1394error_t err;
  dc1394_t* bus_data = dc1394_new();
  if (!bus_data) return 1;
  err = dc1394_camera_enumerate(bus_data, &list);
  DC1394_ERR_RTN(err, "Failed to enumerate cameras");

  printf("Num of connected cameras = %d\n", list->num);
  if (list->num != 2) {
    dc1394_log_error("2 cameras not found!\n");
    return 1;
  }

  bool use_external_trigger = false;
  uint64_t left_guid, right_guid;
  for (int i = 0; i < 2; i++) {
    if (list->ids[i].guid == 13602058368603641)
      left_guid = list->ids[i].guid;
    else if (list->ids[i].guid == 13602058368603640)
      right_guid = list->ids[i].guid;
    else throw 1;
  }
  dc1394_camera_free_list(list);

  cam::CameraCaptureLibdc1394 left_cam(left_guid, bus_data, use_external_trigger);
  cam::CameraCaptureLibdc1394 right_cam(right_guid, bus_data, use_external_trigger);
  cv::Mat left_img, right_img;
  cv::Mat left_rgb_img, right_rgb_img;
  cv::Mat left_disp_img, right_disp_img, left_disp_rawimg;

  if (use_external_trigger)
    SynchronizeCameras(left_cam, right_cam, left_img, right_img);

  std::string save_folder = "/opt/kivan/datasets/calib/basler/";
  //double delta_sum = 0;
  //double tdiff_prev;
  uint64_t i = 0;
  uint64_t save_idx = 0;
  uint64_t left_time = 0, right_time = 0;
  auto start = std::chrono::system_clock::now();
  while (true) {
  //for (; i < 300; i++) {
    printf("\nframe = %lu\n", i);
    left_time = left_cam.Grab(left_img);
    // get right frame closest to left_time
    right_time = right_cam.Grab(right_img);
    //right_time = right_cam.Grab(right_img, left_time);

    printf("left  timestamp = %lu us\n", left_time);
    printf("right timestamp = %lu us\n", right_time);
    double tdiff = static_cast<int>(right_time - left_time) / 1000.0;
    printf("time diff = %.2f ms\n", tdiff);
    //if (i > 0)
    //  delta_sum += (tdiff - tdiff_prev);
    i++;
    //tdiff_prev = tdiff;
    //printf("delta time = %f msec\n", dt_msec);
    //printf("delta sum = %.2f ms\n\n", delta_sum);
    if (left_img.channels() == 2) {
      cv::cvtColor(left_img, left_disp_img, CV_YUV2BGR_UYVY);
      cv::cvtColor(right_img, right_disp_img, CV_YUV2BGR_UYVY);
    }
    else {
      //left_img.copyTo(left_disp_img);
      //right_img.copyTo(right_disp_img);
      ConvertRaw12toRaw8(left_img, left_disp_rawimg);
      ConvertRaw12toRaw8(right_img, right_disp_img);

      // DC1394_BAYER_METHOD_NEAREST=0,
      // DC1394_BAYER_METHOD_SIMPLE,
      // DC1394_BAYER_METHOD_BILINEAR,
      // DC1394_BAYER_METHOD_HQLINEAR,
      // DC1394_BAYER_METHOD_DOWNSAMPLE,
      // DC1394_BAYER_METHOD_EDGESENSE,
      // DC1394_BAYER_METHOD_VNG,
      // DC1394_BAYER_METHOD_AHD

      left_rgb_img.create(left_img.rows, left_img.cols, CV_16UC3);
      dc1394_bayer_decoding_16bit((uint16_t*)left_img.data, (uint16_t*)left_rgb_img.data,
                                  left_img.cols, left_img.rows, DC1394_COLOR_FILTER_RGGB,
                                  DC1394_BAYER_METHOD_BILINEAR, 12);
      ConvertRGB12toBGR8(left_rgb_img, left_disp_img);
      right_rgb_img.create(right_img.rows, right_img.cols, CV_16UC3);
      dc1394_bayer_decoding_16bit((uint16_t*)right_img.data, (uint16_t*)right_rgb_img.data,
                                  right_img.cols, right_img.rows, DC1394_COLOR_FILTER_RGGB,
                                  DC1394_BAYER_METHOD_BILINEAR, 12);
      ConvertRGB12toBGR8(right_rgb_img, right_disp_img);
    }
    cv::imshow("left_image", left_disp_img);
    cv::imshow("left_raw_image", left_disp_rawimg);
    cv::imshow("right_image", right_disp_img);
    int key = cv::waitKey(10);
    if (key == 10) {
      std::stringstream prefix;
      prefix << std::setw(6) << std::setfill('0') << save_idx;
      cv::imwrite(save_folder + "left/" + prefix.str() + "_left.png", left_img);
      cv::imwrite(save_folder + "left/" + prefix.str() + "_left_rgb.png", left_disp_img);
      cv::imwrite(save_folder + "left/" + prefix.str() + "_left_raw.png", left_disp_rawimg);
      cv::imwrite(save_folder + "right/" + prefix.str() + "_right.png", right_img);
      save_idx++;
    }
    else if (key == 27) {
      break;
    }
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
  std::cout << "FPS = " << static_cast<double>(i) / elapsed.count() << "\n";
  
  return 0;
}
