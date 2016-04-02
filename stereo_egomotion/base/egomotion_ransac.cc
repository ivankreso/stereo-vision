#include "egomotion_ransac.h"

#include <array>
#include <random>
#include <unordered_set>
#include <omp.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include "egomotion_solver.h"

#define USE_OMP

namespace egomotion
{

namespace
{

std::vector<int> GetRandomSample(std::uniform_int_distribution<int>& udist,
                                 std::mt19937& rng, size_t N) {
  std::unordered_set<int> set;
  std::vector<int> sample;
  while(sample.size() < N) {
    int rnum = udist(rng);
    if (set.find(rnum) == set.end()) {
      set.insert(rnum);
      sample.push_back(rnum);
    }
  }
  return sample;
}

void AddPointsToSolver(const std::vector<int>& active,
                       const std::vector<Eigen::Vector4d>& pts3d,
                       const std::vector<core::Point>& left_curr,
                       const std::vector<core::Point>& right_curr,
                       EgomotionSolver& solver) {
  solver.ClearPoints();
  for (size_t j = 0; j < active.size(); j++) {
    int idx = active[j];
    solver.AddPoint(Eigen::Vector3d(pts3d[idx][0], pts3d[idx][1], pts3d[idx][2]),
                    left_curr[idx], right_curr[idx]);
  }
}

}

void EgomotionRansac::ProjectToStereo(const Eigen::Vector4d& pt3d,
                                      core::Point& pt_left, core::Point& pt_right) {
  double f = params_.calib.f;
  double cx = params_.calib.cx;
  double cy = params_.calib.cy;
  double b = params_.calib.b;

  double x = pt3d[0] / pt3d[2];
  double y = pt3d[1] / pt3d[2];
  pt_left.x_ = f * x + cx;
  pt_left.y_ = f * y + cy;
  // right camera
  x = (pt3d[0] - b) / pt3d[2];
  pt_right.x_ = f * x + cx;
  pt_right.y_ = pt_left.y_;
}

std::vector<int> EgomotionRansac::GetInliers(const std::vector<Eigen::Vector4d>& pts3d,
                                             const Eigen::Matrix4d Rt,
                                             const std::vector<core::Point>& left_obs,
                                             const std::vector<core::Point>& right_obs) {
  // compute inliers
  std::vector<int> inliers;
  core::Point left_proj, right_proj;
  double square_thr = params_.inlier_threshold * params_.inlier_threshold;
  for (size_t i = 0; i < pts3d.size(); i++) {
    ProjectToStereo(Rt * pts3d[i], left_proj, right_proj);
    double xdiff1 = left_proj.x_ - left_obs[i].x_;
    double ydiff1 = left_proj.y_ - left_obs[i].y_;
    double xdiff2 = right_proj.x_ - right_obs[i].x_;
    double ydiff2 = right_proj.y_ - right_obs[i].y_;
    double square_error = xdiff1*xdiff1 + xdiff2*xdiff2 + ydiff1*ydiff1 + ydiff2*ydiff2;
    //printf("%f < %f\n", square_error, square_thr);
    if (square_error < square_thr)
      inliers.push_back(i);
  }
  return inliers;
}

void EgomotionRansac::PrepareTracks(const track::StereoTrackerBase& tracker,
                                    std::vector<core::Point>& left_prev,
                                    std::vector<core::Point>& left_curr,
                                    std::vector<core::Point>& right_prev,
                                    std::vector<core::Point>& right_curr,
                                    std::vector<int>& active_tracks)
{
  int feats_num = tracker.countFeatures();
  for (int i = 0; i < feats_num; i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    if (feat_left.age_ > 0) {
      track::FeatureInfo feat_right = tracker.featureRight(i);
      left_prev.push_back(feat_left.prev_);
      left_curr.push_back(feat_left.curr_);
      right_prev.push_back(feat_right.prev_);
      right_curr.push_back(feat_right.curr_);
      active_tracks.push_back(i);
    }
  }
}

void EgomotionRansac::UpdateTrackerInliers(const std::vector<int>& active_tracks)
{
  std::vector<bool> dead_tracks(active_tracks.size(), true);
  tracker_inliers_.clear();
  tracker_outliers_.clear();
  for (size_t i = 0; i < inliers_.size(); i++)
    dead_tracks[inliers_[i]] = false;

  for (size_t i = 0; i < dead_tracks.size(); i++) {
    if (dead_tracks[i] == false)
      tracker_inliers_.push_back(active_tracks[i]);
    else
      tracker_outliers_.push_back(active_tracks[i]);
  }
}

bool EgomotionRansac::EstimateMotion(const std::vector<core::Point>& left_prev,
                                     const std::vector<core::Point>& left_curr,
                                     const std::vector<core::Point>& right_prev,
                                     const std::vector<core::Point>& right_curr,
                                     Eigen::Matrix4d& Rt) {
  // return value
  bool success = true;
  // get number of matches
  int N  = left_prev.size();
  if (N < 6)
    return false;
  double cam_params[] = { params_.calib.f, params_.calib.f, params_.calib.cx,
                          params_.calib.cy, params_.calib.b };

  // triangulate 3D points
  //pts3d_ = new double[3 * N];
  std::vector<Eigen::Vector4d> pts3d;
  pts3d.resize(N);
  for (int32_t i = 0; i < N; i++) {
    if(left_prev[i].x_ - right_prev[i].x_ < 0.01) {
      std::cout << "[EgomotionRansac]: Error - negative disparity!\n";
      return false;
    }
    double d = std::max(left_prev[i].x_ - right_prev[i].x_, 0.01);
    //std::cout << d << ": " << p_matched[i].u1p << " - " << p_matched[i].u2p << "\n";
    //if (d <= 0.0) {
    //  std::cout << "[EgomotionLibviso] zero/negative disp: " << d << " -> ";
    //  std::cout << tracks[i].u1p << " - " << tracks[i].u2p << "\n";
    //  throw 1;
    //}
    //else if (d < 1.0) cout << "[EgomotionLibviso] small disp: " << d << "\n";

    pts3d[i][0] = (left_prev[i].x_ - params_.calib.cx) * params_.calib.b / d;
    pts3d[i][1] = (left_prev[i].y_ - params_.calib.cy) * params_.calib.b / d;
    pts3d[i][2] = params_.calib.f * params_.calib.b / d;
    pts3d[i][3] = 1.0;
  }
  
  std::vector<std::vector<int>> active;
  std::vector<std::vector<int>> iter_inliers;
  std::vector<Eigen::Matrix4d> iter_motion;
  active.resize(params_.ransac_iters);
  iter_inliers.resize(params_.ransac_iters);
  iter_motion.resize(params_.ransac_iters);
  // get initial RANSAC estimate
  //omp_set_num_threads(1);
  int ransac_pts = 3;
  std::random_device rd;
  size_t seed = rd();
#ifdef USE_OMP
  #pragma omp parallel
  {
  //std::mt19937 rng(int(time(NULL)) ^ omp_get_thread_num());
  std::mt19937 rng(seed + omp_get_thread_num());
#else
  //std::mt19937 rng(rd());
  std::mt19937 rng(0);
#endif
  std::uniform_int_distribution<int> udist(0, N-1);
  //EgomotionSolver solver(cam_params, false, 1.0);
  // the result is worse if we try robust loss here which makes sense as we really need
  // all 3 points in 2 frames to describe the 3D rigid motion
  EgomotionSolver solver(cam_params, "Squared", 0, params_.use_weighting);

#ifdef USE_OMP
  #pragma omp for
#endif
  for (int i = 0; i < params_.ransac_iters; i++) {
    active[i] = GetRandomSample(udist, rng, ransac_pts);
    // Add points to solver
    AddPointsToSolver(active[i], pts3d, left_curr, right_curr, solver);
    if (!solver.Solve(iter_motion[i])) {
      printf("Solver failed!\n");
    }
    iter_inliers[i] = GetInliers(pts3d, iter_motion[i], left_curr, right_curr);
    // std::cout << "\nInliers = " << iter_inliers[i].size() << "\n";
  }
#ifdef USE_OMP
  }
#endif
  // get the best solution
  int best_iter = 0;
  size_t most_inliers = iter_inliers[0].size();
  for (int i = 1; i < (int)iter_inliers.size(); i++) {
    if (iter_inliers[i].size() > most_inliers) {
      best_iter = i;
      most_inliers = iter_inliers[i].size();
    }
  }
  inliers_ = std::move(iter_inliers[best_iter]);
  //printf("[EgomotionRansac]: RANSAC found most inliers in iter %d / %d\n",
  //       best_iter, params_.ransac_iters);

  //std::cout << iter_motion[best_iter] << "\n";
  // final optimization on all inliers
  EgomotionSolver final_solver(cam_params, params_.loss_function_type, params_.robust_loss_scale,
                               params_.use_weighting);
  final_solver.InitializeParams(iter_motion[best_iter]);
  AddPointsToSolver(inliers_, pts3d, left_curr, right_curr, final_solver);
  if (!final_solver.Solve(Rt)) {
    printf("[EgomotionRansac]: Solver failed!\n");
    return false;
  }

  return success;
}

bool EgomotionRansac::GetMotion(track::StereoTrackerBase& tracker, Eigen::Matrix4d& Rt) {
  // estimate motion
  std::vector<core::Point> left_prev, left_curr, right_prev, right_curr;
  std::vector<int> active_tracks;
  PrepareTracks(tracker, left_prev, left_curr, right_prev, right_curr, active_tracks);

  //Eigen::Matrix4d rt_motion;
  int success = EstimateMotion(left_prev, left_curr, right_prev, right_curr, Rt);
  UpdateTrackerInliers(active_tracks);
  //cv::eigen2cv(rt_motion, Rt);

  return success;
}

}

