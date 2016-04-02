#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <glog/logging.h>

#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"
#include "../../../tracker/base/eval_helper.h"
#include "../../../tracker/stereo/stereo_tracker_orb.h"
#include "../../../tracker/stereo/stereo_tracker_freak.h"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../../tracker/detector/feature_detector_harris_cv.h"
#include "../../../tracker/detector/feature_detector_agast.h"
#include "../../calibration_from_motion/cfm_solver.h"

void RunTracker(const std::string cam_params_file, const std::string dataset_name,
                const std::string left_folder, const std::string right_folder,
                const std::string gt_filepath, size_t start_num, size_t end_num, size_t num_width) {
  double cam_params[5];
  core::FormatHelper::readCameraParams(cam_params_file, cam_params);

  std::vector<cv::Mat> world_motion, camera_motion;
  core::FormatHelper::Read2FrameMotionFromAccCameraMotion(gt_filepath, end_num - start_num,
                                                          world_motion, camera_motion);
  std::cout << "Num of motions = " << world_motion.size() << "\n";

  size_t start = start_num;
  //size_t end_frame = end_num;

  //size_t start = 360;
  //size_t start = 60;
  size_t end_frame = 50;

  //size_t start = 200;
  //size_t end_frame = 220;

  std::stringstream first_filename;
  first_filename << std::setw(num_width) << std::setfill('0') << start << ".png";
  cv::Mat img_left = cv::imread(left_folder + first_filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(right_folder + first_filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_left_prev, img_right_prev;

  //int agast_thr = 20;
  //std::string agast_type = "OAST_9_16";
  //std::shared_ptr<track::FeatureDetectorBase> left_detector =
  //    std::make_shared<track::FeatureDetectorAGAST>(agast_thr, true, agast_type);
  //std::shared_ptr<track::FeatureDetectorBase> right_detector =
  //    std::make_shared<track::FeatureDetectorAGAST>(agast_thr, false, agast_type);

  int block_size = 3;
  std::shared_ptr<track::FeatureDetectorBase> left_detector =
    //std::make_shared<track::FeatureDetectorHarrisCV>(block_size, 3, 0.04, 1e-07, block_size, 20000);
    //std::make_shared<track::FeatureDetectorHarrisCV>(block_size, 3, 0.04, 1e-06, block_size, 20000);
    std::make_shared<track::FeatureDetectorHarrisCV>(block_size, 3, 0.06, 1e-06, block_size, 20000);
  std::shared_ptr<track::FeatureDetectorBase> right_detector = left_detector;

  optim::CFMSolver cfm_solver(img_left.rows, img_left.cols, 0.5);
  //track::StereoTrackerORB tracker(5000, 100);
  //int matching_thr = 20;
  //int max_epipolar_diff = 16;
  int matching_thr = 40;
  int max_epipolar_diff = 0;
  track::StereoTrackerFREAK tracker(left_detector, right_detector, 10000, 100, max_epipolar_diff, 140, 40,
                                    matching_thr, matching_thr, true, 22.0);
  //track::StereoTrackerFREAK tracker(10000, 100, 1, 140, 40, matching_thr, matching_thr,
  //                                  20, "OAST_9_16", false, 22.0);
  tracker.init(img_left, img_right);
  size_t frame_cnt = 0;
  size_t good_cnt = 0;
  auto start_time = std::chrono::system_clock::now();
  for (size_t i = start+1; i <= end_frame; i++) {
    std::cout << "Frame: " << i << " / " << end_frame << "\n";
    cv::swap(img_left, img_left_prev);
    cv::swap(img_right, img_right_prev);
    std::stringstream filename;
    filename << std::setw(num_width) << std::setfill('0') << i << ".png";
    img_left = cv::imread(left_folder + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(right_folder + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
    //if (i == start+1)
    //if (i == start+1 + 16)
    core::MathHelper::PrintMotion(world_motion[i-1]);
    size_t outliers = track::EvalHelper::DrawTracksWithErrors(
        tracker, cam_params, world_motion[i-1], img_left_prev,
        img_right_prev, img_left, img_right, 4.0);
    double outlier_ratio = static_cast<double>(outliers) / tracker.countActiveTracks() * 100.0;
    std::cout << "Num of outliers = " << outliers << " -- "
              << outlier_ratio << "%\n\n";
    if (outlier_ratio < 1.0)
      good_cnt++;

    tracker.track(img_left, img_right);
    cfm_solver.UpdateTracks(tracker, world_motion[i-1]);
    //auto end = std::chrono::system_clock::now();
    std::cout << "Live tracks = " << tracker.countActiveTracks() << "\n";

    //track::EvalHelper::SaveTrackerEvaluation(tracker, cam_params, world_motion[i-start_num-1], 8.0);

    //// TODO - use their calib
    //double error_thr = 5;
    //tracks_passed = track::EvalHelper::FilterOutliersWithGroundtruth(*stereo_tracker,
    //    cam_params, world_motion[i-start_num-1], error_thr); // the best KITTI thresh - 8.0
  }
  std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start_time;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
  std::cout << "Good frames = " << good_cnt << " / " << (end_frame - start) << "\n";

  cfm_solver.Solve();
}

int main(int argc, char** argv) {
  //if (argc != 2)
  //  return 1;
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_v = 0; // 12

  std::string cam_params_file;
  std::string dataset_name;
  std::string left_folder;
  std::string right_folder;
  std::string gt_filepath;
  std::size_t start_num, end_num, num_width;
  core::FormatHelper::ParseKITTIDatasetConfig(argv[1], cam_params_file, dataset_name, left_folder,
                                         right_folder, gt_filepath, start_num, end_num, num_width);

  RunTracker(cam_params_file, dataset_name, left_folder, right_folder, gt_filepath,
             start_num, end_num, num_width);

  return 0;
}

