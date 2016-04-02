#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>

#include <unistd.h>
#include <sys/wait.h>

#include <boost/program_options.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../core/image.h"
#include "../../../core/helper.h"
#include "../../../core/math_helper.h"
#include "../../../stereo_odometry/base/visual_odometry_base.h"
#include "../../../stereo_odometry/base/visual_odometry_ransac.h"
#include "../../../tracker/stereo/experiment_factory.h"
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/stereo/stereo_tracker.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
#include "../../../tracker/refiner/feature_refiner_klt.h"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../../tracker/stereo/tracker_helper.h"
#include "../../../optimization/bundle_adjustment/bundle_adjuster_base.h"
#include "../../../optimization/bundle_adjustment/bundle_adjuster_multiframe.h"
#include "../../../optimization/calibration_bias/deformation_field_solver.h"
#include "../../math_helper.h"
#include "../../../tracker/base/eval_helper.h"
#include "../../../core/format_helper.h"
#include "../../helper_libviso.h"
#include "../../cv_plotter.h"


using namespace core;

void run_visual_odometry(const std::string& source_folder,
                         const std::string& imagelistfn,
                         const std::string& experiment_config,
                         const std::string& cparams_file,
                         const std::string& gt_filepath,
                         const std::string& save_folder_root,
                         const std::string& deformation_field_path)
{
  std::deque<cv::Mat> extr_params; // pose mat
  std::deque<cv::Mat> Rt_params;

  double cam_params[5];
  FormatHelper::readCameraParams(cparams_file, cam_params);

  std::vector<std::string> imagelist;
  bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
  if (!ok || imagelist.empty())
    throw "can not open " + imagelistfn + " or the string list is empty\n";

  int start_frame = 0 * 2;
  //int start_frame = 2 * 2;
  //int start_frame = 228 * 2;
  // 01
  //int start_frame = 431 * 2;
  //int start_frame = 482 * 2;
  int end_frame = imagelist.size();

  cv::Mat img_left_prev, img_right_prev;
  std::cout << source_folder + imagelist[start_frame] << "\n";
  cv::Mat img_left = cv::imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  if (img_left.empty()) {
    std::cout << "Error: no images!\n";
    throw 1;
  }

  std::string output_folder;
  track::FeatureDetectorBase* detector = nullptr;
  track::TrackerBase* mono_tracker = nullptr;
  track::StereoTrackerBase* stereo_tracker = nullptr;
  visodom::VisualOdometryBase* viso = nullptr;
  optim::BundleAdjusterBase* bundle_adjuster = nullptr;
  bool use_ba; // use bundle adjustment
  int img_rows = img_left.rows;
  int img_cols = img_left.cols;
  track::ExperimentFactory::create_experiment(experiment_config, deformation_field_path, cam_params,
                                              img_rows, img_cols, output_folder, &detector, &mono_tracker,
                                              &stereo_tracker, &viso, use_ba, &bundle_adjuster);

  //libviso::Matrix pose = libviso::Matrix::eye(4);
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  //Matrix point_rt = Matrix::eye(4);
  //cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat Rt_inv;
  cv::Mat mat_I = cv::Mat::eye(4, 4, CV_64F);
  mat_I.copyTo(Rt_inv);
  extr_params.push_back(mat_I.clone());
  cv::Mat Rt(4, 4, CV_64F);
  cv::Mat Rt_gt(4, 4, CV_64F);
  cv::Mat Rt_gt_prev = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat Rt_gt_curr = cv::Mat::eye(4, 4, CV_64F);
  core::MathHelper::invTrans(Rt_inv, Rt);
  Rt_params.push_back(Rt.clone());

  // init the tracker
  stereo_tracker->init(img_left, img_right);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<cv::Mat> libviso_Rt_all;
  cv::Mat pose_libviso = cv::Mat::eye(4, 4, CV_64F);
  libviso_Rt_all.push_back(pose_libviso.clone());
  
  std::string track_name;
  bool have_gt = false;
  if (!gt_filepath.empty())
    have_gt = true;
  else throw 1;

  std::string save_folder;
  bool using_kitti = false;
  if (experiment_config.find("tsukuba") != std::string::npos) {
    printf("---------- Using Tsukuba dataset! ------------\n");
    track_name = "00";
    save_folder = save_folder_root + "/";
  }
  else if (experiment_config.find("bb2") != std::string::npos) {
    printf("---------- Using Bumblebee dataset! ------------\n");
    track_name = "bb";
    throw 1;
  }
  // else it is KITTI
  else {
    printf("---------- Using KITTI dataset! ------------\n");
    track_name = imagelistfn.substr(imagelistfn.size()-10,2);
    using_kitti = true;
    save_folder = save_folder_root + "/kitti_" + track_name + "/";
  }
  //std::string save_folder = "/opt/kivan/data/tracker_data/kitti_" + track_name + "/";
  //std::string save_folder = "/mnt/ssd/kivan/datasets/tracker_data/freak/kitti_" + track_name + "/";
  //std::string save_folder = "/mnt/ssd/kivan/datasets/freak_tracker_data/kitti_" + track_name + "/";
  //std::string save_folder = "/opt/kivan/datasets/tracker_data/freak/kitti_" + track_name + "/";
  struct stat st = {0};
  if(stat(save_folder.c_str(), &st) == -1)
    mkdir(save_folder.c_str(), 0700);

  int num_of_motions = ((imagelist.size()/2) - 1);
  std::vector<cv::Mat> gt_world_motion, gt_camera_motion;
  core::FormatHelper::Read2FrameMotionFromAccCameraMotion(gt_filepath, num_of_motions,
                                                          gt_world_motion, gt_camera_motion);

  //int gt_better_cnt = 0;
  auto start = std::chrono::system_clock::now();
  //for (unsigned i = start_frame + 2; i < start_frame+60; i+=2) {
  double max_trans_error = 0.0;
  for (unsigned i = start_frame + 2; i < end_frame; i += 2) {
    int motion_idx = i / 2 - 1;
    std::cout << "motion_idx: " << motion_idx << " / " << (gt_world_motion.size() - 1) << "\n";
    std::cout << source_folder + imagelist[i] << "\n";
    cv::swap(img_left_prev, img_left);
    cv::swap(img_right_prev, img_right);
    img_left = cv::imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);

    stereo_tracker->track(img_left, img_right);

    // filter outliers with GT
    bool gt_filtering_passed = false;
    if (have_gt) {
      std::vector<std::vector<double>> left_tmp, right_tmp;
      gt_filtering_passed = track::EvalHelper::FilterOutliersWithGroundtruth(*stereo_tracker,
                             //cam_params, gt_world_motion[motion_idx], 8.0); // the best thresh - 8.0
                             cam_params, gt_world_motion[motion_idx], 5.0); // the best thresh - 8.0

      //cv::Mat img_lp, img_rp, img_lc, img_rc;
      ////track::EvalHelper::DrawTracksWithBigErrors(*stereo_tracker, cam_params, gt_motion[i/2],
      ////                              4.0, -1, img_left_prev, img_right_prev, img_left, img_right,
      ////                              cv::Scalar(0,0,255), true, img_lp, img_rp, img_lc, img_rc);
      //track::EvalHelper::DrawTracksWithBigErrors(*stereo_tracker, cam_params, gt_motion[i/2],
      //                              6.0, 11, img_left_prev, img_right_prev, img_left, img_right,
      //                              cv::Scalar(0,0,255), true, img_lp, img_rp, img_lc, img_rc);
      //track::EvalHelper::DrawTracksWithBigErrors(*stereo_tracker, cam_params, gt_motion[i/2],
      //                              6.0, 19, img_left_prev, img_right_prev, img_left, img_right,
      //                              cv::Scalar(0,255,0), false, img_lp, img_rp, img_lc, img_rc);
      //track::EvalHelper::DrawTracksWithBigErrors(*stereo_tracker, cam_params, gt_motion[i/2],
      //                              6.0, 30, img_left_prev, img_right_prev, img_left, img_right,
      //                              cv::Scalar(255,0,0), false, img_lp, img_rp, img_lc, img_rc);
      //track::EvalHelper::DrawTracksWithBigErrors(*stereo_tracker, cam_params, gt_motion[i/2],
      //                              6.0, 53, img_left_prev, img_right_prev, img_left, img_right,
      //                              cv::Scalar(0,255,255), false, img_lp, img_rp, img_lc, img_rc);
      //cv::imwrite("left_prev.png", img_lp);
      //cv::waitKey(0);
      //// 11, 19, 30, 53
      //track::EvalHelper::DrawTrackPatches(*stereo_tracker, 53, 15, img_lp, img_rp, img_lc, img_rc);
    }
    //continue;

    // run odometry optimization
    visodom::VisualOdometryRansac* viso_ransac = dynamic_cast<visodom::VisualOdometryRansac*>(viso);
    // Rt is the motion of world points with respect to the camera
    Rt = viso->getMotion(*stereo_tracker);

    if (!Rt.empty()) {
      // is it better to kill or not to kill outliers in tracker?
      std::vector<int> inliers = viso->getTrackerInliers();
      std::vector<int> outliers = viso->getTrackerOutliers();
      std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      //FeatureHelper::FilterOutlierTracks(*stereo_tracker, active_tracks, inliers, 30.0, 0.1);
      //FeatureHelper::FilterRansacOutliers(*stereo_tracker, active_tracks, inliers);
      for (size_t j = 0; j < outliers.size(); j++)
        stereo_tracker->removeTrack(outliers[j]);
      std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      std::cout << "Serializing to disk\n";
      // save data to archive
      std::ofstream ofs(save_folder + "tracks_" + std::to_string(motion_idx));
      //boost::archive::text_oarchive oarchive(ofs);
      boost::archive::binary_oarchive oarchive(ofs);
      // write class instance to archive
      oarchive << *(dynamic_cast<track::StereoTracker*>(stereo_tracker));
      // archive and stream closed when destructors are called

      //libviso::Matrix Rt_libviso = viso->getMotion();
      //MathHelper::matrixToMat(Rt_libviso, Rt);
      if (have_gt)
        std::cout << "groundtruth:\n" << gt_world_motion[motion_idx] << std::endl;
      std::cout << "two-frame odometry:\n" << Rt << std::endl;

      std::cout << "-----------------------------------------------------------------------\n\n";
    } else {
      std::cout << "libviso ... failed!" << "\n";
      return;
      //throw "Error\n";
      //extr_params.push_back(Rt_inv.clone());
      //exit(1);
    }
  }
  auto end = std::chrono::system_clock::now();
  //auto elapsed = end - start;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " secs\n";

  delete detector;
  delete mono_tracker;
  delete stereo_tracker;
  delete viso;
}


int main(int argc, char** argv)
{
  std::string config_file;
  std::string experiment_config;
  std::string imagelistfn;
  std::string cam_params_file;
  std::string source_folder;
  std::string gt_filepath;
  std::string deformation_field_path;  
  std::string save_folder;

  if (argc != 7){
    std::cout << "Usage:\n" << argv[0] << " -c dataset_config -e experiment_config -s save_folder\n";
    return 0;
  }

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help, h", "produce help message")
      ("config,c", po::value<std::string>(&config_file), "config filename")
      ("experiment_config,e", po::value<std::string>(&experiment_config), "experiment config filename")
      ("save_folder,s", po::value<std::string>(&save_folder), "save folder");
    po::options_description cmdline_options;
    cmdline_options.add(generic);

    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
    if (vm.count("help")) {
      std::cout << generic;
      return 0;
    }

    po::options_description config("Config file options");
    config.add_options()
      ("camera_params,p", po::value<std::string>(&cam_params_file)->default_value("camera_params.txt"), "camera params file")
      ("source_folder,s", po::value<std::string>(&source_folder), "folder with source")
      ("imglist,l", po::value<std::string>(&imagelistfn), "file with image list")
      ("groundtruth,g", po::value<std::string>(&gt_filepath), "file with motion GT")
      ("deformation_field,d", po::value<std::string>(&deformation_field_path), "file with deformation field");
    std::ifstream ifs(config_file);
    if (!ifs) {
      throw "can not open config file: " + config_file + "\n";
    }
    else {
      po::store(parse_config_file(ifs, config, true), vm);
      notify(vm);
    }
  }
  catch(std::exception& e) {
    std::cout << e.what() << "\n";
    return -1;
  }

  std::cout << "Using track config = " << config_file << '\n';
  run_visual_odometry(source_folder, imagelistfn, experiment_config, cam_params_file, gt_filepath,
                      save_folder, deformation_field_path);

  return 0;
}
