#ifndef __OPTIMIZATION_SBA_BASE__
#define __OPTIMIZATION_SBA_BASE__

#include <opencv2/core/core.hpp>

#include "../../core/types.h"

namespace optim {

class SBAbase
{
 public:
  enum BAType { kMotion, kStructureAndMotion };

  virtual ~SBAbase() {}
  virtual void setCameraIntrinsics(const cv::Mat& cam_params) = 0;
  virtual void addCameraMotion(const cv::Mat& Rt) = 0;
  virtual void addPoint(const cv::Mat& pt, double weight) = 0;
  virtual void addMonoProj(int ci, int pi, const core::Point& proj) = 0;
  virtual void addStereoProj(int ci, int pi, const core::Point& proj_left, const core::Point& proj_right) = 0;
  virtual void runSBA() = 0;
  
  // get camera extrinsic transform from cam local coord to world coord (1-frame coord)
  virtual cv::Mat getCameraRt(int ci) const = 0;

  //virtual void setCameraTrans(int ci, Eigen::Matrix4d& Rt) = 0;
};

}

#endif
