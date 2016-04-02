#ifndef STEREO_EGOMOTION_BASE_EGOMOTION_BASE_
#define STEREO_EGOMOTION_BASE_EGOMOTION_BASE_

#include <Eigen/Core>
#include "../../tracker/stereo/stereo_tracker_base.h"

namespace egomotion
{

class EgomotionBase
{
 public:
  virtual ~EgomotionBase() {}
  virtual bool GetMotion(track::StereoTrackerBase& tracker, Eigen::Matrix4d& Rt) = 0;
  virtual std::vector<int> GetTrackerInliers() = 0;
  virtual std::vector<int> GetTrackerOutliers() = 0;
};

}

#endif
