#ifndef OPTIMIZATION_BUNDLE_ADJUSTER_MULTIFRAME_
#define OPTIMIZATION_BUNDLE_ADJUSTER_MULTIFRAME_

//#include <vector>
#include <deque>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "bundle_adjuster_base.h"
#include "sba_base.h"
#include "../../tracker/stereo/stereo_tracker_base.h"

namespace optim {

  //typedef pair<char*, int> Key;
  //typedef map< Key , char*> Mapa;

class BundleAdjusterMultiframe : public BundleAdjusterBase
{
 public:
  BundleAdjusterMultiframe(int sba_frames, int max_feats, SBAbase::BAType ba_type, bool use_weighting);

  virtual void set_camera_params(const double* cam_params);
  virtual void update_tracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt);

  virtual void optimize();
  virtual cv::Mat camera_motion(int cam_num);
  virtual cv::Mat camera_motion_acc(int cam_num);
  virtual int camera_motion_num() { return camera_motion_.size(); }

 private:
  void initTracks(double (&cam_params)[5]);

  int frame_cnt_ = 0;
  int sba_frames_;
  int max_feats_;
  SBAbase::BAType ba_type_;
  bool use_weighting_;

  std::deque<std::vector<core::Point>> left_tracks_;
  std::deque<std::vector<core::Point>> right_tracks_;
  std::deque<std::vector<core::Point>> first_left_tracks_;
  std::deque<std::vector<core::Point>> first_right_tracks_;

  std::deque<std::vector<cv::Mat>> pts3d_;

  cv::Mat cam_intr_;
  std::deque<std::vector<int>> tracks_age_;
  std::deque<cv::Mat> cam_extr_;
  std::vector<cv::Mat> camera_motion_;
  std::vector<cv::Mat> camera_motion_acc_;
  std::vector<cv::Mat> world_motion_;
};

}

#endif
