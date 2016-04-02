#ifndef OPTIMIZATION_BUNDLE_ADJUSTER_MFI_
#define OPTIMIZATION_BUNDLE_ADJUSTER_MFI_

#include <array>
#include <unordered_map>
#include <tuple>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "bundle_adjuster_base.h"
#include "sba_base.h"
#include "../../tracker/stereo/stereo_tracker_base.h"

namespace optim {

class BundleAdjusterMFI : public BundleAdjusterBase
{
 public:
  BundleAdjusterMFI(SBAbase::BAType ba_type, bool use_weighting);

  virtual void set_camera_params(const double* cam_params);
  virtual void update_tracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt);
  virtual void optimize();
  virtual cv::Mat camera_motion(int cam_num) { return camera_motion_[cam_num].clone(); }
  virtual cv::Mat camera_motion_acc(int cam_num);
  virtual int camera_motion_num() { return camera_motion_.size(); }

 private:
  void init_tracks(double (&cam_params)[5]);
  void integrate_feature(const cv::Mat& rt, const int age,
                         std::array<std::tuple<core::Point,core::Point>,3>& tracks);

  int frame_cnt_ = 0;
  SBAbase::BAType ba_type_;
  bool use_weighting_;

  std::unordered_map<int, std::array<std::tuple<core::Point,core::Point>,3>> tracks_map_;
  std::unordered_map<int, int> age_map_;

  cv::Mat cam_intr_;
  std::vector<cv::Mat> camera_motion_;
  std::vector<cv::Mat> pts_motion_;
  std::vector<cv::Mat> camera_motion_acc_;
  cv::Mat init_motion_;
};

inline
cv::Mat BundleAdjusterMFI::camera_motion_acc(int cam_num)
{
  if(cam_num >= (int)camera_motion_acc_.size())
    throw 1;
  return camera_motion_acc_[cam_num].clone();
}

}

#endif
