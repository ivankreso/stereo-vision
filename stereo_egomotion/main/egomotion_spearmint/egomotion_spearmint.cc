#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

//#include <unistd.h>
//#include <sys/wait.h>

#include <boost/python.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../core/helper.h"
#include "../../../core/math_helper.h"
#include "../../../stereo_egomotion/base/egomotion_base.h"
#include "../../../stereo_egomotion/base/evaluator.h"
#include "../../../tracker/stereo/experiment_factory.h"
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../../optimization/bundle_adjustment/bundle_adjuster_base.h"
//#include "../../../tracker/stereo/tracker_helper.h"
//#include "../../math_helper.h"
#include "../../../tracker/base/eval_helper.h"
#include "../../../tracker/base/tracker_helper.h"
#include "../../../core/format_helper.h"


using namespace core;

void RunEgomotion(const std::string& cam_params_file, const std::string& dataset_name,
                  const std::string& left_folder, const std::string& right_folder,
                  size_t start_num, size_t end_num,
                  size_t num_width, std::shared_ptr<track::StereoTrackerBase>& stereo_tracker,
                  std::vector<Eigen::Matrix4d>& egomotion_poses) {
  double cam_params[5];
  FormatHelper::readCameraParams(cam_params_file, cam_params);

  //Eigen::VectorXd camera_params(5);
  //for (int i = 0; i < 5; i++)
  //  camera_params[i] = cam_params[i];
  bool debug = false;
  //debug = true;

  std::string output_folder = ".";
  cv::Mat img_left_prev, img_right_prev;

  std::stringstream first_filename;
  first_filename << std::setw(num_width) << std::setfill('0') << start_num << ".png";
  cv::Mat img_left = cv::imread(left_folder + first_filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(right_folder + first_filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
  if (img_left.empty()) {
    std::cout << "Error: no images!\n";
    throw 1;
  }
  bool use_ba = false; // use bundle adjustment
  
  std::shared_ptr<optim::BundleAdjusterBase> bundle_adjuster;
  std::shared_ptr<egomotion::EgomotionBase> egomotion;
  int ransac_iters = 1000;
  double ransac_threshold = 1.5;
  std::string loss_function_type = "Squared";
  double robust_loss_scale = 0;
  bool use_weighting = true;

  egomotion::EgomotionRansac::Parameters params;
  params.ransac_iters = ransac_iters;
  params.inlier_threshold = ransac_threshold;
  params.loss_function_type = loss_function_type;
  params.robust_loss_scale = robust_loss_scale;
  params.use_weighting = use_weighting;
  params.calib.f = cam_params[0];
  params.calib.cx = cam_params[2];
  params.calib.cy = cam_params[3];
  params.calib.b = cam_params[4];
  egomotion = std::make_shared<egomotion::EgomotionRansac>(params);

  if (use_ba) {
    throw 1;
    //bundle_adjuster->SetCameraParams(camera_params);
  }

  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Rt = Eigen::Matrix4d::Identity();

  // init the tracker
  stereo_tracker->init(img_left, img_right);

  // matrix transforms points in current camera coordinate system to world coord system
  egomotion_poses.push_back(pose);

  //std::vector<cv::Mat> gt_world_motion, gt_camera_motion;
  //if (have_gt) {
  //  int num_of_motions = ((imagelist.size()/2) - 1);
  //  core::FormatHelper::Read2FrameMotionFromAccCameraMotion(gt_filepath, num_of_motions,
  //                                                          gt_world_motion, gt_camera_motion);
  //}
  int frame_num = 0;
  auto start = std::chrono::system_clock::now();
  for (size_t i = start_num+1; i <= end_num; i++) {
    frame_num++;
    std::cout << "\rImage num: " << i << " / " << end_num << "\n";
    //if (have_gt)
    //  std::cout << "motion_idx: " << motion_idx << " / " << (gt_world_motion.size() - 1) << "\n";
    cv::swap(img_left_prev, img_left);
    cv::swap(img_right_prev, img_right);

    std::stringstream filename;
    filename << std::setw(num_width) << std::setfill('0') << i << ".png";
    img_left = cv::imread(left_folder + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(right_folder + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

    stereo_tracker->track(img_left, img_right);

    // default prior
    double min_disp = 0.01;
    double max_disp_diff = 40.0;
    // filter tracks with prior knowledge
    track::TrackerHelper::FilterTracksWithPriorOld(*stereo_tracker, max_disp_diff, min_disp);

    // draw tracks before ransac
    if (debug)
      track::EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
                                          "all_left", "all_right");

    // run egomotion optimization
    // Rt is the motion of world points with respect to the camera
    bool success = egomotion->GetMotion(*stereo_tracker, Rt);

    if (success) {
      // filter outlier tracks
      //std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      std::vector<int> outliers = egomotion->GetTrackerOutliers();
      for (size_t j = 0; j < outliers.size(); j++)
        stereo_tracker->removeTrack(outliers[j]);
      //std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";

      pose = pose * Rt.inverse();
      egomotion_poses.push_back(pose);

      //std::cout << "Two-frame odometry:\n" << Rt << std::endl;
      if (use_ba) {
        bundle_adjuster->UpdateTracks(*stereo_tracker, Eigen::Matrix4d::Identity());
        if (frame_num >= (bundle_adjuster->num_frames() - 1)) {
          if(!bundle_adjuster->Optimize())
            return;
          if (use_ba) {
            std::cout << "BA odometry:\n" << bundle_adjuster->camera_motions().back() << "\n";
          }
        }
      }
      //std::cout << "-----------------------------------------------------------------------\n\n";
    } else {
      std::cout << "Egomotion failed!\n";
      throw 1;
    }
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "[Egomotion]: elapsed time = " << elapsed.count() << " sec\n";

  //std::cout << "Writing track " << dataset_name << " to files\n";
  //std::ofstream egomotion_ofile(output_folder + '/' + dataset_name + ".txt");

  //std::ofstream ba_ofile;
  //std::ofstream ba_ofile2;
  ////cv::Mat sba_pose;
  //if (use_ba) {
  //  std::string ba_folder = output_folder.substr(0, output_folder.size()-1) + "_ba/";
  //  ba_ofile.open(ba_folder + dataset_name + ".txt");
  //}
  //pose = Eigen::Matrix4d::Identity();
  //std::vector<Eigen::Matrix4d> bundle_adjusted_motions;
  //if (use_ba) {
  //  bundle_adjusted_motions = bundle_adjuster->camera_motions();
  //  if (egomotion_poses.size() != bundle_adjusted_motions.size()+1) {
  //    std::cout << "[egomotion.cc] " << egomotion_poses.size() << "!="
  //              << bundle_adjusted_motions.size()+1 << "\n";
  //    return;
  //  }
  //}
}

double run_orb(const std::string config_file, int patch_size, int num_levels, float scale_factor,
               int max_dist_stereo, int max_dist_mono) {
  std::string cam_params_file;
  std::string dataset_name;
  std::string left_folder;
  std::string right_folder;
  std::string gt_filepath;
  std::size_t start_num, end_num, num_width;

  core::FormatHelper::ParseKITTIDatasetConfig(config_file, cam_params_file, dataset_name,
                                              left_folder, right_folder, gt_filepath, start_num,
                                              end_num, num_width);
  std::vector<Eigen::Matrix4d> poses;
  std::shared_ptr<track::StereoTrackerBase> stereo_tracker;
  bool status = track::ExperimentFactory::CreateValidationORB(
      patch_size, num_levels, scale_factor, max_dist_stereo, max_dist_mono,
      &stereo_tracker);
  if (!status) {
    std::cout << "Invalid experiment config!\n";
    throw 1;
  }
  RunEgomotion(cam_params_file, dataset_name, left_folder, right_folder, start_num, end_num,
               num_width, stereo_tracker, poses);
  double trans_error, rot_error;
  egomotion::Evaluator::Eval(gt_filepath, poses, trans_error, rot_error);
  return trans_error;

  return 0;
}

double run_agast_freak(const std::string config_file, int agast_threshold, std::string agast_type,
                       std::string normalize_orientation, float freak_size,
                       int max_xdiff, int max_dist_stereo, int max_dist_mono) {
  std::string cam_params_file;
  std::string dataset_name;
  std::string left_folder;
  std::string right_folder;
  std::string gt_filepath;
  std::size_t start_num, end_num, num_width;
  bool norm_orient = false;
  if (normalize_orientation == "YES")
    norm_orient = true;

  core::FormatHelper::ParseKITTIDatasetConfig(config_file, cam_params_file, dataset_name,
                                              left_folder, right_folder, gt_filepath, start_num,
                                              end_num, num_width);
  std::vector<Eigen::Matrix4d> poses;
  std::shared_ptr<track::StereoTrackerBase> stereo_tracker;
  bool status = track::ExperimentFactory::CreateValidationAgastFreak(
      agast_threshold, agast_type, norm_orient, freak_size, max_xdiff,
      max_dist_stereo, max_dist_mono, &stereo_tracker);
  if (!status) {
    std::cout << "Invalid experiment config!\n";
    throw 1;
  }
  RunEgomotion(cam_params_file, dataset_name, left_folder, right_folder, start_num,
               end_num, num_width, stereo_tracker, poses);
  double trans_error, rot_error;
  egomotion::Evaluator::Eval(gt_filepath, poses, trans_error, rot_error);
  return trans_error;

  return 0;
}


double run_harris_freak(const std::string config_file, const std::string normalize_orientation,
                        const float freak_size, const int max_xdiff, const int max_dist_stereo,
                        const int max_dist_mono) {
  const int block_sz = 3;
  const int filter_sz = 3;
  const double k = 0.04;
  const double thr = 1e-07;
  const int margin = block_sz;
  const int max_features = 20000;
  const int max_tracks = 10000;

  std::string cam_params_file;
  std::string dataset_name;
  std::string left_folder;
  std::string right_folder;
  std::string gt_filepath;
  std::size_t start_num, end_num, num_width;
  bool norm_orient = false;
  if (normalize_orientation == "YES")
    norm_orient = true;

  core::FormatHelper::ParseKITTIDatasetConfig(config_file, cam_params_file, dataset_name,
      left_folder, right_folder, gt_filepath, start_num, end_num, num_width);
  std::vector<Eigen::Matrix4d> poses;
  std::shared_ptr<track::StereoTrackerBase> stereo_tracker;
  bool status = track::ExperimentFactory::CreateValidationHarrisFreak(
      block_sz, filter_sz, k, thr, margin, max_features, max_tracks, norm_orient,
      freak_size, max_xdiff, max_dist_stereo, max_dist_mono, &stereo_tracker);
  if (!status) {
    std::cout << "Invalid experiment config!\n";
    throw 1;
  }
  RunEgomotion(cam_params_file, dataset_name, left_folder, right_folder, start_num,
               end_num, num_width, stereo_tracker, poses);
  double trans_error, rot_error;
  egomotion::Evaluator::Eval(gt_filepath, poses, trans_error, rot_error);
  return trans_error;

  return 0;
}

BOOST_PYTHON_MODULE(libegomotion) {
  boost::python::def("run_orb", run_orb);
  boost::python::def("run_harris_freak", run_harris_freak);
  boost::python::def("run_agast_freak", run_agast_freak);
}
