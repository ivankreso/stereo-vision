#include "bundle_adjuster.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

namespace optim {

BundleAdjuster::BundleAdjuster(int number_of_frames, int max_features, std::string loss_type,
                               std::vector<double> loss_params, bool use_weighting) :
      max_features_(max_features), loss_type_(loss_type), loss_params_(loss_params),
      use_weighting_(use_weighting) {
  if (number_of_frames > 6) {
    std::cout << "Error: to much frames for BA!\n";
    throw 1;
  }
  frame_cnt_ = 0;
  *mutable_num_frames() = number_of_frames;
  num_motions_ = num_frames() - 1;
  // add first cam location - I matrix
  camera_motion_acc_.push_back(Eigen::Matrix4d::Identity());

  curr_idx_.assign(max_features_, -1);
}

// rt is the motion of points with respect to the camera
void BundleAdjuster::UpdateTracks(const track::StereoTrackerBase& tracker,
                                   const Eigen::Matrix4d& world_rt) {
  frame_cnt_++;
  init_world_motion_.push_back(world_rt);

  // all tracks are now older by one
  for (auto& elem : tracks_)
    elem.second.dist_from_cframe++;

  for (int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    int age = feat_left.age_;
    if(age > 0) {
      // if the track is newly added we need to save the previous point also
      if(age == 1) {
        TrackData data;
        data.dist_from_cframe = 0;
        data.left_tracks.push_back(feat_left.prev_);
        data.right_tracks.push_back(feat_right.prev_);
        for (int j = 0; ; j++) {
          auto key = std::make_tuple(i,j);
          if (tracks_.find(key) == tracks_.end()) {
            curr_idx_[i] = j;
            tracks_[key] = data;
            break;
          }
        }
      }
      // now add the tracked point in current frame
      auto key = std::make_tuple(i, curr_idx_[i]);
      //if (tracks_.find(key) == tracks_.end())
      TrackData& data = tracks_.at(key);
      // reset the distance counter
      data.dist_from_cframe = 0;
      data.left_tracks.push_back(feat_left.curr_);
      data.right_tracks.push_back(feat_right.curr_);
    }
  }

  // this doesn't work with unordered_map !!!
  //for (const auto& elem : tracks_) {
  //  if (elem.second.dist_from_cframe >= num_motions_)
  //    tracks_.erase(elem.first);
  //}
  // clean tracks which dropped outside BA window
  for (auto it = std::begin(tracks_); it != std::end(tracks_);) {
    if (it->second.dist_from_cframe >= (num_motions_))
      it = tracks_.erase(it);
    else
      it++;
  }
}

bool BundleAdjuster::Optimize() {
  assert(init_world_motion_.size() > 1);
  if (camera_params().size() != 5) {
    std::cout << "[BundleAdjuster]: Wrong camera params size!\n";
    return false;
  }
  BundleAdjustmentSolver solver(loss_type_, loss_params_, use_weighting_);

  // set cam intrinsics
  solver.SetCameraParams(camera_params());

  // first add motions
  int start_motion = init_world_motion_.size() - num_motions_;
  for (size_t i = start_motion; i < init_world_motion_.size(); i++)
    solver.AddCameraMotion(init_world_motion_[i]);

  // iterate through tracks and add them to solver
  for(const auto& pair : tracks_) {
    const TrackData& data = pair.second;
    solver.AddTrackData(data);
  }

  bool status = solver.Solve();
  if (!status) {
    std::cout << "[BundleAdjuster] No CONVERGENCE!\n";
    return false;
  }

  if (camera_motions().size() == 0) {
    //Eigen::Matrix4d cam_acc = Eigen::Matrix4d::Identity();
    for (int i = 0; i < num_motions_; i++)
      mutable_camera_motions().push_back(solver.GetCameraMotion(i));
      //Eigen::Matrix4d camera_rt;
      //Eigen::Matrix4d world_rt = solver.GetCameraMotion(i);
      //core::MathHelper::InverseTransform(world_rt, camera_rt);
      //cam_acc = cam_acc * camera_rt;
      //camera_motion_acc_.push_back(cam_acc);
  }
  else {
    mutable_camera_motions().push_back(solver.GetCameraMotion(num_motions_ - 1));

    //Eigen::Matrix4d camera_rt;
    //Eigen::Matrix4d world_rt = solver.GetCameraMotion(num_motions_ - 1);
    //core::MathHelper::InverseTransform(world_rt, camera_rt);
    //std::cout << world_rt.inverse() << "\n==\n" << camera_rt << "\n";
    //std::cout << camera_rt * world_rt << "\n";
    //std::cout << world_rt.inverse() * world_rt << "\n";
    //cv::Mat cv_rt, cv_rt_inv;
    //cv::eigen2cv(world_rt, cv_rt);
    //core::MathHelper::invTrans(cv_rt, cv_rt_inv);
    //std::cout << cv_rt << "\n==\n" << cv_rt_inv << "\n";
    //std::cout << cv_rt_inv * cv_rt << "\n";
    //std::cout << cv_rt.inv() * cv_rt << "\n";

    //mutable_camera_motions().push_back(camera_rt);
    //camera_motion_acc_.push_back(camera_motion_acc_.back() * camera_rt);
  }
  return true;

  //if (camera_motions().size() == 1) {
  //  //cv::Mat cam_acc = cv::Mat::eye(4, 4, CV_64F);
  //  Eigen::Matrix4d cam_acc = Eigen::Matrix4d::Identity();
  //  for (int i = 0; i < num_motions_; i++) {
  //    Eigen::Matrix4d camera_rt;
  //    Eigen::Matrix4d world_rt = solver.GetCameraMotion(i);
  //    core::MathHelper::InverseTransform(world_rt, camera_rt);
  //    mutable_camera_motions().push_back(camera_rt);
  //    cam_acc = cam_acc * camera_rt;
  //    camera_motion_acc_.push_back(cam_acc);
  //    //world_motion_.push_back(rt);
  //  }
  //}
  //else {
  //  Eigen::Matrix4d camera_rt;
  //  Eigen::Matrix4d world_rt = solver.GetCameraMotion(num_motions_ - 1);
  //  core::MathHelper::InverseTransform(world_rt, camera_rt);
  //  std::cout << world_rt << "\n==\n" << camera_rt << "\n";
  //  mutable_camera_motions().push_back(camera_rt);
  //  camera_motion_acc_.push_back(camera_motion_acc_.back() * camera_rt);
  //  //world_motion_.push_back(rt);
  //}
}

}
