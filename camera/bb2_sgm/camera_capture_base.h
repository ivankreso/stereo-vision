#ifndef CAMERA_CAMERA_CAPTURE_BASE_
#define CAMERA_CAMERA_CAPTURE_BASE_

#include <opencv2/core/core.hpp>

namespace cam {

class CameraCaptureBase
{
public:
  virtual ~CameraCaptureBase() {}
  virtual uint64_t Grab(cv::Mat& image) = 0;
  virtual uint64_t Grab(cv::Mat& image, uint64_t time) = 0;
};

}

#endif
