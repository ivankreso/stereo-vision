#include "bundle_adjuster_mfi.h"

#include "sba_base.h"
#include "../../core/math_helper.h"
#include "sba_ceres.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace optim {

BundleAdjusterMFI::BundleAdjusterMFI(SBAbase::BAType ba_type, bool use_weighting)
    : ba_type_(ba_type), use_weighting_(use_weighting)
{
  frame_cnt_ = 0;
  // add first cam location - I matrix
  camera_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));
  camera_motion_acc_.push_back(cv::Mat::eye(4, 4, CV_64F));
  pts_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));
}

// call only with tracks in first frame
void BundleAdjusterMFI::set_camera_params(const double* cam_params)
{
  cam_intr_ = cv::Mat::zeros(5, 1, CV_64F);
  for(int i = 0; i < 5; i++)
    cam_intr_.at<double>(i) = cam_params[i];
}

// rt is the motion of points with respect to the camera
void BundleAdjusterMFI::update_tracks(const track::StereoTrackerBase& tracker, const cv::Mat& rt)
{
  assert(!cam_intr_.empty());
  frame_cnt_++;

  // in first frame we can't run multi-frame optimization
  rt.copyTo(init_motion_);
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
      auto& tracks = tracks_map_[i];
      age_map_[i] = age;
      if(age == 1) {
        // [] will also clear any old tracks on that location
        tracks[0] = std::make_tuple(feat_left.prev_, feat_right.prev_);
        tracks[1] = std::make_tuple(feat_left.prev_, feat_right.prev_);
      }
      // else integrate the feature from 3 frames ago that is about to be removed
      else {
        integrate_feature(pts_motion_.back(), age-1, tracks);
        // new prev = previous curr
        tracks[1] = std::make_tuple(feat_left.prev_, feat_right.prev_);
        //tracks[1] = tracks[2];
      }
      // now add the tracked point in current frame
      assert(tracks_map_.find(i) != tracks_map_.end());
      assert(age_map_.find(i) != age_map_.end());
      tracks[2] = std::make_tuple(feat_left.curr_, feat_right.curr_);
    }
    // if the track just died -> delete it
    else {
      auto iter = tracks_map_.find(i);
      if(iter != tracks_map_.end())
        tracks_map_.erase(iter);
    }
  }
}

void BundleAdjusterMFI::integrate_feature(const cv::Mat& rt, const int age,
                                          std::array<std::tuple<core::Point,core::Point>,3>& tracks)
{
  // integrated previous feature
  auto& integ_prev = tracks[0];
  core::Point& integ_left = std::get<0>(integ_prev);
  //std::cout << std::get<0>(integ_prev) << " before\n";
  core::Point& integ_right = std::get<1>(integ_prev);
  // observed previous feature
  auto& obs_prev = tracks[1];
  core::Point& obs_left = std::get<0>(obs_prev);
  core::Point& obs_right = std::get<1>(obs_prev);

  cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
  // predict observed feature
  core::MathHelper::triangulate(cam_intr_, obs_left, obs_right, pt3d);
  pt3d = rt * pt3d;
  core::Point reproj_obs_left, reproj_obs_right;
  core::MathHelper::project_stereo(cam_intr_, pt3d, reproj_obs_left, reproj_obs_right);

  // predict integrated feature
  core::MathHelper::triangulate(cam_intr_, integ_left, integ_right, pt3d);
  pt3d = rt * pt3d;
  core::Point reproj_integ_left, reproj_integ_right;
  core::MathHelper::project_stereo(cam_intr_, pt3d, reproj_integ_left, reproj_integ_right);

  // get new integrated features
  integ_left.x_ = (reproj_obs_left.x_ + age * reproj_integ_left.x_) / (double)(1 + age);
  integ_left.y_ = (reproj_obs_left.y_ + age * reproj_integ_left.y_) / (double)(1 + age);
  integ_right.x_ = (reproj_obs_right.x_ + age * reproj_integ_right.x_) / (double)(1 + age);
  integ_right.y_ = (reproj_obs_right.y_ + age * reproj_integ_right.y_) / (double)(1 + age);
  //std::cout << std::get<0>(integ_prev) << " after\n";
}

void BundleAdjusterMFI::optimize()
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
  // iterate through points
  for(const auto& pair : tracks_map_) {
    //  std::cout << "adding pt3d: \n" << pts3d_[start_frame][i] << "\n";
    //std::cout << "start frame: " << start_frame << "\n" << cam_poses[start_frame] << "\n";
    const auto& tracks = pair.second;
    core::Point curr_left = std::get<0>(tracks[2]);
    core::Point curr_right = std::get<1>(tracks[2]);

    int age = age_map_[pair.first];

    //double init_weight = 1.0 / (double)((stereo_tracks.size() - 1) - first_frame);
    double cp_x = cam_intr_.at<double>(2);
    for(size_t i = 0; i < 2; i++) {
      if(i == 0 && age == 1)
        continue;
      double weight = 1.0;
      if(i == 0) {
       weight = (double)(age - 1);
      }
      core::Point prev_left = std::get<0>(tracks[i]);
      core::Point prev_right = std::get<1>(tracks[i]);
      // weight the points
      if(use_weighting_)
        weight = weight + 1.0/(std::fabs(prev_left.x_ - cp_x)/std::fabs(cp_x) + 0.05); // slightly better
      //  //weight = weight/(std::fabs(prev_left.x_ - cp_x)/std::fabs(cp_x) + 0.05); // slightly better
      //  weight = 1.0/(std::fabs(prev_left.x_ - cp_x)/std::fabs(cp_x) + 0.05); // slightly better
      //  //weight = 1.0/(std::fabs(last_left.x_ - cp_x)/std::fabs(cp_x) + 0.05);   // same as Libviso
      ////std::cout << weight << "\n";

      //std::cout << "i = " << i << "\n";
      //std::cout << "previous:" << prev_left << "\n" << prev_right << "\n";
      //std::cout << "current:" << curr_left << "\n" << curr_right << "\n";
      cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
      core::MathHelper::triangulate(cam_intr_, prev_left, prev_right, pt3d);
      // move the points from its frame to the second last (previous frame)
      // add the points to BA problem structure
      sba->addPoint(pt3d, weight);
      sba->addStereoProj(0, pt_idx++, curr_left, curr_right);
    }
  }

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
