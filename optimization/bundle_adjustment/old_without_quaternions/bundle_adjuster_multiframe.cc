#include "bundle_adjuster_multiframe.h"

#include "sba_base.h"
#include "../../core/math_helper.h"
#include "sba_ceres.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

//namespace
//{
//cv::Mat DrawPoint(const core::Point& pt, const cv::Mat& img)
//{
//  cv::Mat color_img;
//  cv::cvtColor(img, color_img, cv::COLOR_GRAY2RGB);
//  cv::Scalar color(0,0,255);
//  cv::Point cvpt;
//  cvpt.x = pt.x_;
//  cvpt.y = pt.y_;
//  cv::circle(color_img, cvpt, 3, color, -1, 8);
//  return color_img;
//}
//}

namespace optim {

BundleAdjusterMultiframe::BundleAdjusterMultiframe(int sba_frames, int max_feats, SBAbase::BAType ba_type,
                                                   bool use_weighting)
                                   : sba_frames_(sba_frames), max_feats_(max_feats), ba_type_(ba_type),
                                     use_weighting_(use_weighting)
{
  frame_cnt_ = 0;
  // add first cam location - I matrix
  camera_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));
  camera_motion_acc_.push_back(cv::Mat::eye(4, 4, CV_64F));
  world_motion_.push_back(cv::Mat::eye(4, 4, CV_64F));

  //images_left_.push_back(img_left);
  //images_right_.push_back(img_right);
}

// call only with tracks in first frame
void BundleAdjusterMultiframe::set_camera_params(const double* cam_params)
{
  cam_intr_ = cv::Mat::zeros(5, 1, CV_64F);
  for(int i = 0; i < 5; i++)
    cam_intr_.at<double>(i) = cam_params[i];
}

// Rt is the motion of world points with respect to camera
void BundleAdjusterMultiframe::update_tracks(const track::StereoTrackerBase& tracker, const cv::Mat& Rt)
{
  assert(!cam_intr_.empty());
  frame_cnt_++;

  std::vector<core::Point> pts_left, pts_right;
  std::vector<core::Point> pts_left_first, pts_right_first;
  pts_left_first.resize(max_feats_);
  pts_right_first.resize(max_feats_);
  //std::vector<Eigen::Vector4d> points3d;
  std::vector<cv::Mat> points3d;
  std::vector<int> age(max_feats_, 0);
  pts_left.resize(max_feats_);
  pts_right.resize(max_feats_);

  cam_extr_.push_back(Rt.clone());
  //Eigen::Matrix4d cam_extr;
  //cv2eigen(Rt, cam_extr);
  //cam_extr_.push_back(cam_extr);
  //Eigen::VectorXd cam_intr(cam_params);
  //Eigen::Matrix<double,5,1> cam_intr(cam_params);
  //assert(cam_intr.rows() == 5 && cam_intr.cols() == 1);
  //cam_intr_.push_back(cam_intr);

  //cv::Mat cam_intr(5, 1, CV_64F, cam_params);
  //assert(cam_intr.rows == 5 && cam_intr.cols == 1);
  //cam_intr_.push_back(cam_intr.clone());

  assert(max_feats_ == tracker.countFeatures());

  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    age[i] = feat_left.age_;
    cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
    if(age[i] > 0) {
      pts_left[i] = feat_left.curr_;
      pts_right[i] = feat_right.curr_;
      // triangulate into 3D only the valid tracks previous points
      core::MathHelper::triangulate(cam_intr_, feat_left.prev_, feat_right.prev_, pt3d);

      // we need to save the previous point also
      pts_left_first[i] = feat_left.prev_;
      pts_right_first[i] = feat_right.prev_;
    }
    else {
      pts_left_first[i] = core::Point(-1000, -1000);
      pts_right_first[i] = core::Point(-1000, -1000);
    }
    points3d.push_back(pt3d.clone());
  }

  left_tracks_.push_back(pts_left);
  right_tracks_.push_back(pts_right);
  first_left_tracks_.push_back(pts_left_first);
  first_right_tracks_.push_back(pts_right_first);
  pts3d_.push_back(points3d);
  tracks_age_.push_back(age);
  //images_left_.push_back(img_left);
  //images_right_.push_back(img_right);

  // clear oldest frame data
  if(frame_cnt_ >= sba_frames_) {
    left_tracks_.pop_front();
    right_tracks_.pop_front();
    first_left_tracks_.pop_front();
    first_right_tracks_.pop_front();
    tracks_age_.pop_front();
    pts3d_.pop_front();
    cam_extr_.pop_front();
    //images_left_.pop_front();
    //images_right_.pop_front();
  }
}

void BundleAdjusterMultiframe::optimize()
{
  BALProblemStereo sba_problem(false);
  SBAbase* sba;
  sba = new SBAceres(&sba_problem, ba_type_);

  //if(!((int)cam_extr_.size() == sba_frames_
  //   && (int)tracks_age_.size() == sba_frames_
  //   && (int)pts3d_.size() == sba_frames_
  //   && (int)left_tracks_.size() == sba_frames_
  //   && (int)right_tracks_.size() == sba_frames_
  //   && (int)first_left_tracks_.size() == sba_frames_
  //   && (int)first_right_tracks_.size() == sba_frames_))
  //    throw 1;

  // set cam intrinsics
  sba->setCameraIntrinsics(cam_intr_);
  // first add cams/frames
  cv::Mat cam_rt;
  cv::Mat world_rt = cv::Mat::eye(4,4,CV_64F);
  std::vector<cv::Mat> cam_poses;
  cam_poses.push_back(world_rt.clone());
  for(size_t i = 0; i < cam_extr_.size(); i++) {
    //cam_pose = cam_pose * cam_extr_[i];
    //cam_poses.push_back(cam_pose.clone());
    world_rt = cam_extr_[i] * world_rt;
    //core::MathHelper::invTrans(cam_pose, cam_pose_inv);
    sba->addCameraMotion(world_rt);
    core::MathHelper::invTrans(world_rt, cam_rt);
    cam_poses.push_back(cam_rt.clone());
    //std::cout << "Adding motion in frame " << i << ":\n" << cam_pose_inv << "\n\n";
    //sba->addCam(cam_pose.inv(), cam_intr_[i+1]);
    //std::cout << cam_pose.inv() << "\n\n" << cam_pose_inv << "\n\n\n";
  }
  //for(size_t i = 0; i < cam_poses.size(); i++)
  //  std::cout << cam_poses[i] << "\n";
  //throw 1;

  // add point projections in frames
  // we need this pt_idx to keep track of the current visible point index
  int pt_idx = 0;
  int num_obs = 0;
  int num_disp_drops = 0;
  // iterate through points
  for(size_t i = 0; i < left_tracks_[0].size(); i++) {
    // we can have multiple tracks on one index i
    int start_frame = 0;
    // first count frames where points are visible
    std::vector<int> start_frames;
    std::vector<int> num_of_frames;
    //while(true) {
    //int vframes = 1;
    int vframes = 0;
    int prev_age = 0;
    for(size_t j = start_frame; j < tracks_age_.size(); j++) {
      if(tracks_age_[j][i] > prev_age) {
        //std::cout << i << " -- " << j << " age = " << tracks_age_[j][i] << '\n';
        prev_age = tracks_age_[j][i];
        vframes++;
        //if(vframes == 2)
        if(vframes == 1)
          start_frame = j;
        if(j < (tracks_age_.size()-1))
          continue;
      }

      //// --- use this to ignore all tracks not visible in first frame
      //if(vframes > 0) {
      //  start_frames.push_back(start_frame);
      //  num_of_frames.push_back(vframes);
      //}
      //break;
      //// ---

      // --- use this if you want to use all tracks
      // if switching to new track then decrement j
      else if(tracks_age_[j][i] == 1)
        j--;
      
      // FIREWALL - take only points that are not to old...
      // take all points... need to check triangulation in world system in SBA algorithm
      // here we save the start position and number of frames for each track
      //if(vframes > 1) {
      if(vframes > 0) {
        //std::cout << "adding: " << i << " start_frame = " << start_frame << " frames = " << vframes << '\n';
        start_frames.push_back(start_frame);
        num_of_frames.push_back(vframes);
        // reset init state
        //start_frame += (vframes-1);
        //vframes = 1;
        vframes = 0;
        prev_age = 0;
      }
      // ---
    }
    //}

    for(size_t l = 0; l < start_frames.size(); l++) {
      // add triangulated points in fist camera frame
      //sba->addPoint(pts3d_[0][i]);

      //if(start_frame > 0)
      //  std::cout << "adding pt3d: \n" << pts3d_[start_frame][i] << "\n";
      start_frame = start_frames[l];
      vframes = num_of_frames[l];
      //if(start_frames.size() > 1) {
      //  for(size_t k = 0; k < tracks_age_.size(); k++)
      //    std::cout << "Age " << k << " = " << tracks_age_[k][i] << "\n";
      //  std::cout << "\npt index: " << i << "\nstart frame: " << start_frame << "\nvisible in frames: " << vframes << "\n";
      //  //throw "Error";
      //}
      //std::cout << "start frame: " << start_frame << "\n" << cam_poses[start_frame] << "\n";
      double weight = 1.0;
      if(use_weighting_) {
        double cp_x = cam_intr_.at<double>(2);
        if(first_left_tracks_[start_frame][i].x_ < 0.0) {
          std::cout << first_left_tracks_[start_frame][i];
          throw 1;
        }
        weight = 1.0/(std::fabs(first_left_tracks_[start_frame][i].x_ - cp_x)/std::fabs(cp_x) + 0.05);
      }
      sba->addPoint(cam_poses[start_frame] * pts3d_[start_frame][i], weight);
      //sba->addPoint(pts3d_[start_frame][i]);
      // TODO: can we add points in newer frames?
      double disp_prev = 0.0;
      core::Point left_pt, right_pt;
      //for(int j = start_frame; j < (start_frame + vframes); j++) {
      for(int j = start_frame; j < (start_frame + vframes); j++) {
        //if(j == start_frame) {
        //  left_pt = first_left_tracks_[j][i];
        //  right_pt = first_right_tracks_[j][i];
        //}
        //else {
        //  left_pt = left_tracks_[j-1][i];
        //  right_pt = right_tracks_[j-1][i];
        //}
        assert(j < left_tracks_.size());
        left_pt = left_tracks_[j][i];
        right_pt = right_tracks_[j][i];
        double disp = left_pt.x_ - right_pt.x_;
        //assert(disp >= disp_prev);
        num_obs++;
        //if(disp < disp_prev) {
        if(j != start_frame && std::abs(disp_prev - disp) > 10.0) {
          num_disp_drops++;
          printf("\33[0;31m [BundleAdjusterMultiframe]: BIG disp dropping: %f -> %f !\33[0m\n", disp_prev, disp);
          std::cout << "Frame = " << j << " - feature: " << i << "\n";
          //std::cout << left_tracks_[j-1][i] << " -- " << right_tracks_[j-1][i] << "\n";
          std::cout << left_pt << " -- " << right_pt << "\n";
          if(disp < 0.0)
            throw "Error\n";
        }
        if(left_pt.x_ < 0.0 || right_pt.x_ < 0.0 ||
           left_pt.x_ > 1500.0 || right_pt.x_ > 1500.0 ||
           left_pt.y_ < 0.0 || right_pt.y_ < 0.0 ||
           left_pt.y_ > 1000.0 || right_pt.y_ > 1000.0)
        {
          std::cout << "[BundleAdjusterMultiframe]: Error proj range!\n";
          std::cout << left_pt << " -- " << right_pt << "\n";
          throw "Error\n";
        }
        disp_prev = disp;
        if(disp < 0.1) std::cout << "[BundleAdjusterMultiframe]: small disp: " << disp << std::endl;
        //std::cout << "[dbg] adding in cam = " << j << " point idx = " << pt_idx << " -> i = " << i << ":  ";
        //std::cout << left_tracks_[j][i] << " - " << right_tracks_[j][i] << " : d = " << disp << "\n";
        sba->addStereoProj(j, pt_idx, left_pt, right_pt);

        //cv::Mat img_left = DrawPoint(left_tracks_[j][i], images_left_[j]);
        //cv::imshow(std::to_string(j)+"_left", img_left);
        //cv::Mat img_right = DrawPoint(right_tracks_[j][i], images_right_[j]);
        //cv::imshow(std::to_string(j) + "_right", img_right);
      }
      //cv::waitKey(0);
      //cv::destroyAllWindows();

      //std::cout << "\n";
      pt_idx++;
    }
  }
  std::cout << "disp_drops / num of obs = " << num_disp_drops << " / " << num_obs << "\n";

  //std::cout << "Rt before:\n";
  //for(int i = 0; i < sba_frames_-1; i++) {
  //  cv::Mat Rt = sba->getCameraRt(i);
  //  std::cout << Rt << std::endl;
  //}

  sba->runSBA();

  //std::cout << "Rt before:\n";
  //for(size_t i = 0; i < cam_extr_.size(); i++)
  //  std::cout << cam_extr_[i] << std::endl;

  //std::cout << "Rt after:\n";
  //for(int i = 0; i < sba_frames_-1; i++) {
  //  //Eigen::Matrix4d Rt = sba->getCameraRt(i);
  //  cv::Mat Rt = sba->getCameraRt(i);
  //  std::cout << Rt << std::endl;
  //}

  // save the best solution
  // TODO: when to choose the final Rt solution?

  int rt_size = camera_motion_.size();
  assert(rt_size >= sba_frames_ || rt_size == 1);
  cv::Mat wrt_prev = cv::Mat::eye(4, 4, CV_64F);
  if(rt_size == 1) {
    for(size_t i = 0; i < cam_extr_.size(); i++) {
      cv::Mat wrt_next = sba->getCameraRt(i);
      cv::Mat wrt_prev_inv;
      core::MathHelper::invTrans(wrt_prev, wrt_prev_inv);
      // rt_23 = rt_13 * rt_12^-1
      world_motion_.push_back(wrt_next * wrt_prev_inv);
      cv::Mat camera_rt;
      core::MathHelper::invTrans(world_motion_.back(), camera_rt);
      camera_motion_.push_back(camera_rt.clone());
      camera_motion_acc_.push_back(camera_motion_acc_.back() * camera_rt);
      //std::cout << camera_motion_.back() << "\n";
      wrt_next.copyTo(wrt_prev);
    }
  }
  // variant 1: rt_23 = rt_12_2F^-1 *  rt_13_ba
  else {
    int rt_size = camera_motion_.size();
    assert(rt_size >= sba_frames_);

    int last_motion = cam_extr_.size() - 1;
    cv::Mat wrt_prev = sba->getCameraRt(last_motion - 1);
    cv::Mat wrt_last = sba->getCameraRt(last_motion);
    cv::Mat wrt_prev_inv;
    core::MathHelper::invTrans(wrt_prev, wrt_prev_inv);
    world_motion_.push_back(wrt_last * wrt_prev_inv);
    cv::Mat camera_rt;
    core::MathHelper::invTrans(world_motion_.back(), camera_rt);
    camera_motion_.push_back(camera_rt.clone());
    camera_motion_acc_.push_back(camera_motion_acc_.back() * camera_rt);
    //std::cout << camera_motion_.back() << "\n";
  }

  // bad idea -> This doesn't work on tsukuba
  //// variant 2: rt_23 = rt_12_MF^-1 *  rt_13_ba
  //else {
  //  int last_motion = cam_extr_.size() - 1;
  //  cv::Mat wrt_last = sba->getCameraRt(last_motion);
  //  int w_num = world_motion_.size();
  //  for(int i = (cam_extr_.size()-1); i > 0; i--)
  //    wrt_prev = world_motion_[w_num - i] * wrt_prev;

  //  cv::Mat wrt_prev_inv;
  //  core::MathHelper::invTrans(wrt_prev, wrt_prev_inv);
  //  world_motion_.push_back(wrt_last * wrt_prev_inv);
  //  cv::Mat camera_rt;
  //  core::MathHelper::invTrans(world_motion_.back(), camera_rt);
  //  camera_motion_.push_back(camera_rt.clone());
  //  camera_motion_acc_.push_back(camera_motion_acc_.back() * camera_rt);
  //  //std::cout << camera_motion_.back() << "\n";
  //}

  delete sba;
}

cv::Mat BundleAdjusterMultiframe::camera_motion(int cam_num)
{
  if(cam_num >= (int)camera_motion_.size())
    throw 1;
  return camera_motion_[cam_num].clone();
}

cv::Mat BundleAdjusterMultiframe::camera_motion_acc(int cam_num)
{
  if(cam_num >= (int)camera_motion_.size())
    throw 1;
  return camera_motion_acc_[cam_num].clone();
  //cv::Mat motion = cv::Mat::eye(4, 4, CV_64F);
  //for(int i = 0; i <= cam_num; i++)
  //  motion *= camera_motion_[i];
  //return motion;
}

}
