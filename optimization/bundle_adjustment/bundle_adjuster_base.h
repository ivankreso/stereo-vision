#ifndef OPTIMIZATION_BUNDLE_ADJUSTMENT_BUNDLE_ADJUSTER_BASE_H_
#define OPTIMIZATION_BUNDLE_ADJUSTMENT_BUNDLE_ADJUSTER_BASE_H_

#include <vector>
#include <Eigen/Core>
#include "../../tracker/stereo/stereo_tracker_base.h"

namespace optim {

class BundleAdjusterBase {
 public:
  virtual ~BundleAdjusterBase() {}
  virtual void UpdateTracks(const track::StereoTrackerBase& tracker,
                            const Eigen::Matrix4d& Rt) = 0;
  virtual bool Optimize() = 0;

  void SetCameraParams(const Eigen::VectorXd& camera_params) {
    camera_params_ = camera_params;
  }

  const std::vector<Eigen::Matrix4d>& camera_motions() const {
    return camera_motions_;
  }
  std::vector<Eigen::Matrix4d> mutable_camera_motions() const {
    return camera_motions_;
  }
  int num_frames() const {
    return num_frames_;
  }
  int* mutable_num_frames() {
    return &num_frames_;
  }

 protected:
  std::vector<Eigen::Matrix4d>& mutable_camera_motions() {
    return camera_motions_;
  }
  Eigen::VectorXd& camera_params() {
    return camera_params_;
  }

 private:
  int num_frames_;
  std::vector<Eigen::Matrix4d> camera_motions_;
  //std::vector<Eigen::Matrix4d> camera_motions_cumulative;
  Eigen::VectorXd camera_params_;
};

}   // namespace optim

#endif  // OPTIMIZATION_BUNDLE_ADJUSTMENT_BUNDLE_ADJUSTER_BASE_H_
