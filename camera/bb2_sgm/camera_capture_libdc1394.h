#ifndef CAMERA_CAMERA_CAPTURE_LIBDC1394_
#define CAMERA_CAMERA_CAPTURE_LIBDC1394_

#include "camera_capture_base.h"

#include <opencv2/core/core.hpp>
#include <dc1394/dc1394.h>

namespace cam {

class CameraCaptureLibdc1394 : public CameraCaptureBase
{
 public:
  CameraCaptureLibdc1394(uint64_t cam_guid, dc1394_t* bus_data, bool use_external_trigger);
  ~CameraCaptureLibdc1394();
  uint64_t Grab(cv::Mat* left_img, cv::Mat* right_img);
  // grab newest frame in the buffer and flush the buffer
  virtual uint64_t Grab(cv::Mat& image);
  // grab frame closest to given time and flush all frames older then that time
  virtual uint64_t Grab(cv::Mat& image, uint64_t time);

 private:
  uint32_t width_, height_;
  uint64_t cam_guid_;
  uint32_t ring_buffer_size_;
  bool use_external_trigger_;
  dc1394camera_t* camera_;
  dc1394video_mode_t video_mode_;

  std::vector<dc1394video_frame_t*> frames_;

};

}

#endif
