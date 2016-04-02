#ifndef OPTIMIZATION_BUNDLE_ADJUSTER_H_
#define OPTIMIZATION_BUNDLE_ADJUSTER_H_

#include <unordered_map>
#include <tuple>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "bundle_adjustment_solver.h"
#include "bundle_adjuster_base.h"
#include "../../core/math_helper.h"
#include "../../tracker/stereo/stereo_tracker_base.h"

namespace optim {

typedef std::tuple<int,int> TrackKey;

struct KeyHash : public std::unary_function<TrackKey, std::size_t>
{
  std::size_t operator()(const TrackKey& k) const
  {
    return std::get<0>(k) ^ std::get<1>(k);
  }
};

struct KeyEqual : public std::binary_function<TrackKey, TrackKey, bool>
{
  bool operator()(const TrackKey& v0, const TrackKey& v1) const
  {
    return (std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1));
  }
};

typedef std::unordered_map<TrackKey, TrackData, KeyHash, KeyEqual> tracker_map;

class BundleAdjuster : public BundleAdjusterBase
{
 public:
  BundleAdjuster(int num_frames, int max_features, std::string loss_type,
                 std::vector<double> loss_params, bool use_weighting);

  void UpdateTracks(const track::StereoTrackerBase& tracker,
                    const Eigen::Matrix4d& world_rt) override;
  bool Optimize() override;

  //virtual cv::Mat camera_motion(int cam_num) { return camera_motion_[cam_num].clone(); }
  //virtual cv::Mat camera_motion_acc(int cam_num) { return camera_motion_acc_[cam_num].clone(); }
  //virtual int camera_motion_num() { return camera_motion_.size(); }
  //virtual int num_frames() { return num_frames_; }

 private:
  void InitTracks(double (&cam_params)[5]);

  int frame_cnt_ = 0;
  int max_features_;
  //int num_frames_;
  int num_motions_;
  bool use_weighting_;
  std::string loss_type_;
  std::vector<double> loss_params_;

  tracker_map tracks_;
  std::vector<int> curr_idx_;

  std::vector<Eigen::Matrix4d> init_world_motion_;
  std::vector<Eigen::Matrix4d> pts_motion_;
  std::vector<Eigen::Matrix4d> camera_motion_acc_;
  std::vector<Eigen::Matrix4d> twoframe_pts_motion_;
  Eigen::Matrix4d init_motion_;
};

}

#endif
