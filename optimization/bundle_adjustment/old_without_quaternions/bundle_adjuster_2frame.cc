#include "bundle_adjuster_2frame.h"

#include "sba_base.h"
#include "../../core/math_helper.h"
#include "sba_ceres.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace optim {

BundleAdjuster2frame::BundleAdjuster2frame(int nframes_ba, SBAbase::BAType ba_type, bool use_weighting)
    : nframes_ba_(nframes_ba), ba_type_(ba_type), use_weighting_(use_weighting)
{
  frame_cnt_ = 0;
  // add first cam location - I matrix
  camera_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));
  camera_motion_acc_.push_back(cv::Mat::eye(4, 4, CV_64F));
  pts_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));
  twoframe_pts_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));
}

// call only with tracks in first frame
void BundleAdjuster2frame::set_camera_params(const double* cam_params)
{
  cam_intr_ = cv::Mat::zeros(5, 1, CV_64F);
  for(int i = 0; i < 5; i++)
    cam_intr_.at<double>(i) = cam_params[i];
}

// rt is the motion of points with respect to the camera
void BundleAdjuster2frame::update_tracks(const track::StereoTrackerBase& tracker, const cv::Mat& rt)
{
  assert(!cam_intr_.empty());
  frame_cnt_++;

  rt.copyTo(init_motion_);
  twoframe_pts_motion_.push_back(rt.clone());
  if(camera_motion_.size() == 1) {
    pts_motion_.push_back(rt.clone());
    cv::Mat rt_inv;
    core::MathHelper::invTrans(rt, rt_inv);
    camera_motion_.push_back(rt_inv);
    camera_motion_acc_.push_back(rt_inv);
  }

  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    int age = feat_left.age_;
    if(age > 0) {
      // if the track is newly added we need to save the previous point also
      if(age == 1) {
        auto track_data = std::make_tuple(feat_left.prev_, feat_right.prev_, 0);
        std::vector<std::tuple<core::Point,core::Point,int>> stereo_tracks;
        stereo_tracks.push_back(track_data);
        // [] will also clear any old tracks on that location
        tracks_map_[i] = stereo_tracks;
        //assert(tracks_map_[i].size() == 1);
      }
      // now add the tracked point in current frame
      assert(tracks_map_.find(i) != tracks_map_.end());
      auto track_data = std::make_tuple(feat_left.curr_, feat_right.curr_, age);
      tracks_map_[i].push_back(track_data);
    }
    // if the track just died -> delete it
    else {
      auto iter = tracks_map_.find(i);
      if(iter != tracks_map_.end())
        tracks_map_.erase(iter);
    }
  }
}

void BundleAdjuster2frame::optimize()
{
  assert(camera_motion_.size() > 1);
  BALProblemStereo sba_problem(false);
  SBAbase* sba;
  sba = new SBAceres(&sba_problem, ba_type_);

  // set cam intrinsics
  sba->setCameraIntrinsics(cam_intr_);

  // first add last motion
  sba->addCameraMotion(init_motion_);

  // add point projections in frames
  // we need this pt_idx to keep track of the current visible point index
  int pt_idx = 0;
  int num_disp_drops = 0;
  // iterate through points
  for(const auto& pair : tracks_map_) {
    //  std::cout << "adding pt3d: \n" << pts3d_[start_frame][i] << "\n";
    //std::cout << "start frame: " << start_frame << "\n" << cam_poses[start_frame] << "\n";
    const auto& stereo_tracks = pair.second;
    core::Point last_left = std::get<0>(stereo_tracks.back());
    core::Point last_right = std::get<1>(stereo_tracks.back());

    int age_prev = std::get<2>(stereo_tracks[0]);
    assert(age_prev == 0);
    for(size_t i = 1; i < stereo_tracks.size(); i++) {
      int age = std::get<2>(stereo_tracks[i]);
      if(age != (age_prev+1)) {
        std::cout << "age prev = " << age_prev << "\n";
        std::cout << "age = " << age << "\n";
        throw 1;
      }
      age_prev = age;
    }

    double disp_prev;
    //for(size_t i = 0; i < (stereo_tracks.size() - 1); i++) {
    size_t first_frame = static_cast<size_t>(std::max(0, static_cast<int>(stereo_tracks.size()) - nframes_ba_));

    //double init_weight = 1.0 / (double)((stereo_tracks.size() - 1) - first_frame); // bad idea on tsukuba
    //double init_weight = 1.0;
    double cp_x = cam_intr_.at<double>(2);
    for(size_t i = first_frame; i < (stereo_tracks.size() - 1); i++) {
    //for(size_t i = first_frame; i < (first_frame + 1); i++) {
      core::Point curr_left = std::get<0>(stereo_tracks[i]);
      core::Point curr_right = std::get<1>(stereo_tracks[i]);
      if(i == first_frame)
        disp_prev = curr_left.x_ - curr_right.x_;

      // weight the points
      //double weight = 1.0;
      double weight = 1.0;
      if(use_weighting_)
        //weight = init_weight/(std::fabs(curr_left.x_ - cp_x)/std::fabs(cp_x) + 0.05); // better ?
        weight = 1.0/(std::fabs(curr_left.x_ - cp_x)/std::fabs(cp_x) + 0.05); // slightly better
        //weight = 1.0/(std::fabs(last_left.x_ - cp_x)/std::fabs(cp_x) + 0.05);   // same as Libviso

      cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
      core::MathHelper::triangulate(cam_intr_, curr_left, curr_right, pt3d);
      // move the points from its frame to the second last (previous frame)
      int frames_to_move = stereo_tracks.size() - i - 2;
      //std::cout << "init_pos = " << "\n" << pt3d << "\n\n";
      for(int j = frames_to_move; j > 0; j--) {
        assert(j <= pts_motion_.size());
        int cam_idx = pts_motion_.size() - j;
        //std::cout << "cam_idx = " << cam_idx << "\n" << pts_motion_[cam_idx] << "\n\n";
        // TODO: use 2-frame motion or BA motion
        //pt3d = pts_motion_[cam_idx] * pt3d;
        pt3d = twoframe_pts_motion_[cam_idx] * pt3d; // mali
        //std::cout << "cam_idx = " << cam_idx << "\n" << pt3d << "\n\n";
      }
      // add the points to BA problem structure
      sba->addPoint(pt3d, weight);
      sba->addStereoProj(0, pt_idx++, last_left, last_right);

      double disp = curr_left.x_ - curr_right.x_;
      if(disp < 0.0) throw "Error\n";
      if(disp < 0.1)
        printf("\33[0;31m [BundleAdjustFast]: small disp: %f !\33[0m\n", disp);
      //if(std::abs(disp_prev - disp) > 10.0) {
      if(i > first_frame && std::abs(disp_prev - disp) > 8.0) {
        num_disp_drops++;
        printf("\33[0;31m [BundleAdjustFast]: BIG disp dropping: %f -> %f !\33[0m\n", disp_prev, disp);
        //std::cout << "Frame = " << j << " - feature: " << i << "\n";
        //std::cout << left_tracks_[j-1][i] << " -- " << right_tracks_[j-1][i] << "\n";
        //std::cout << left_pt << " -- " << right_pt << "\n";
      }
      disp_prev = disp;
    }

  }
  printf("[BundleAdjustFast]: disp_drops / num of obs = %d / %d\n", num_disp_drops, pt_idx);

  //std::cout << "Rt before:\n";
  //for(int i = 0; i < sba_frames_; i++) {
  //  cv::Mat Rt = sba->getCameraRt(i);
  //  std::cout << Rt << std::endl;
  //}

  sba->runSBA();

  //std::cout << "Rt before:\n";
  //for(size_t i = 0; i < cam_extr_.size(); i++)
  //  std::cout << cam_extr_[i] << std::endl;

  //std::cout << "Rt after:\n";
  //for(int i = 0; i < sba_frames_; i++) {
  //  //Eigen::Matrix4d Rt = sba->getCameraRt(i);
  //  cv::Mat Rt = sba->getCameraRt(i);
  //  std::cout << Rt << std::endl;
  //}

  // add the result as final solution
  cv::Mat rt_world = sba->getCameraRt(0);
  pts_motion_.push_back(rt_world.clone());
  cv::Mat rt_cam;
  core::MathHelper::invTrans(rt_world, rt_cam);
  camera_motion_.push_back(rt_cam.clone());
  camera_motion_acc_.push_back(camera_motion_acc_.back() * rt_cam);

  delete sba;
}

}
