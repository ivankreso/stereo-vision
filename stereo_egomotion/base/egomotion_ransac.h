#ifndef STEREO_EGOMOTION_EGOMOTION_RANSAC_
#define STEREO_EGOMOTION_EGOMOTION_RANSAC_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include "../../tracker/stereo/stereo_tracker_base.h"
#include "egomotion_base.h"

namespace egomotion
{

class EgomotionRansac : public EgomotionBase
{
 public:
  struct CalibParams {
    double f;
    double cx;
    double cy;
    double b;
  };
  struct Parameters {
    int ransac_iters;
    double inlier_threshold;
    std::string loss_function_type;
    double robust_loss_scale;
    CalibParams calib;
    bool use_weighting;
  };

  EgomotionRansac(Parameters& params) : params_(params) {}
  bool GetMotion(track::StereoTrackerBase& tracker, Eigen::Matrix4d& Rt) override;

  virtual std::vector<int> GetTrackerInliers() { return tracker_inliers_; }
  virtual std::vector<int> GetTrackerOutliers() { return tracker_outliers_; }

 private:
  void UpdateTrackerInliers(const std::vector<int>& active_tracks);
  void PrepareTracks(const track::StereoTrackerBase& tracker,
                     std::vector<core::Point>& left_prev,
                     std::vector<core::Point>& left_curr,
                     std::vector<core::Point>& right_prev,
                     std::vector<core::Point>& right_curr,
                     std::vector<int>& active_tracks);
  bool EstimateMotion(const std::vector<core::Point>& left_prev,
                      const std::vector<core::Point>& left_curr,
                      const std::vector<core::Point>& right_prev,
                      const std::vector<core::Point>& right_curr,
                      Eigen::Matrix4d& Rt);
  std::vector<int> GetInliers(const std::vector<Eigen::Vector4d>& pts3d,
                              const Eigen::Matrix4d Rt,
                              const std::vector<core::Point>& left_obs,
                              const std::vector<core::Point>& right_obs);
  void ProjectToStereo(const Eigen::Vector4d& pt3d,
                       core::Point& pt_left, core::Point& pt_right);

  std::vector<int> inliers_;
  std::vector<int> tracker_inliers_;
  std::vector<int> tracker_outliers_;
  Parameters params_;
};

}

#endif
