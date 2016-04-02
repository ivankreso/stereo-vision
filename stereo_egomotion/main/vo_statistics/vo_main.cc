#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

#include <unistd.h>
#include <sys/wait.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../core/image.h"
#include "../../../core/helper.h"
#include "../../../stereo_odometry/base/visual_odometry_base.h"
#include "../../../stereo_odometry/base/visual_odometry_ransac.h"
#include "../../../tracker/stereo/experiment_factory.h"
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
#include "../../../tracker/refiner/feature_refiner_klt.h"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../../tracker/stereo/tracker_helper.h"
#include "../../../optimization/bundle_adjustment/bundle_adjuster_base.h"
#include "../../../optimization/bundle_adjustment/bundle_adjuster_multiframe.h"
#include "../../math_helper.h"
#include "../../../tracker/base/eval_helper.h"
#include "../../../core/format_helper.h"
#include "../../helper_libviso.h"
#include "../../cv_plotter.h"

using namespace core;
using namespace vo;
using namespace track;
using namespace std;

void run_visual_odometry(const std::string& source_folder,
                         const std::string& imagelistfn,
                         const std::string& experiment_config,
                         const std::string& cparams_file,
                         const std::string& gt_filepath)
{
  std::deque<cv::Mat> extr_params; // pose mat
  std::deque<cv::Mat> Rt_params;

  double cam_params[5];
  FormatHelper::readCameraParams(cparams_file, cam_params);

  std::vector<std::string> imagelist;
  bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
    throw "can not open " + imagelistfn + " or the string list is empty\n";

  int start_frame = 0 * 2;
  // 01
  //int start_frame = 431 * 2;
  //int start_frame = 482 * 2;
  int end_frame = imagelist.size();
  //int nframes = (imagelist.size() - start_frame) / 2;
  bool debug = false;
  //debug = true;

  cv::Mat img_left_prev, img_right_prev;
  std::cout << source_folder + imagelist[start_frame] << "\n";
  cv::Mat img_left = cv::imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  if(img_left.empty()) {
    std::cout << "Error: no images!\n";
    throw 1;
  }

  std::string output_folder;
  FeatureDetectorBase* detector = nullptr;
  TrackerBase* mono_tracker = nullptr;
  StereoTrackerBase* stereo_tracker = nullptr;
  visodom::VisualOdometryBase* viso = nullptr;
  //optim::FeatureHelperSBA* bundle_adjuster = nullptr;
  optim::BundleAdjusterBase* bundle_adjuster = nullptr;
  bool use_ba; // use bundle adjustment
  //track::ExperimentFactory::create_experiment(experiment_config, "/home/kivan/Dropbox/experiment_data/04_deformation_field_matrix.yml", cam_params,
  //                                            img_left.rows, img_left.cols, output_folder, &detector,
  //                                            &mono_tracker, &stereo_tracker, &viso, use_ba, &bundle_adjuster);
  track::ExperimentFactory::create_experiment(experiment_config, "", cam_params,
                                              0, 0, output_folder, &detector,
                                              &mono_tracker, &stereo_tracker, &viso, use_ba, &bundle_adjuster);

  //libviso::Matrix pose = libviso::Matrix::eye(4);
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  //Matrix point_rt = Matrix::eye(4);
  //cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat Rt_inv;
  cv::Mat mat_I = cv::Mat::eye(4, 4, CV_64F);
  mat_I.copyTo(Rt_inv);
  cv::Vec<double,7> trans_vec;
  extr_params.push_back(mat_I.clone());
  cv::Mat Rt(4, 4, CV_64F);
  cv::Mat Rt_gt(4, 4, CV_64F);
  cv::Mat Rt_gt_prev = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat Rt_gt_curr = cv::Mat::eye(4, 4, CV_64F);
  core::MathHelper::invTrans(Rt_inv, Rt);
  Rt_params.push_back(Rt.clone());

  cv::Mat prev_location_viso = cv::Mat::zeros(4, 1, CV_64F);

  // init the tracker
  stereo_tracker->init(img_left, img_right);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<cv::Mat> libviso_Rt_all;
  cv::Mat pose_libviso = cv::Mat::eye(4, 4, CV_64F);
  libviso_Rt_all.push_back(pose_libviso.clone());

  std::string track_name;
  bool have_gt = false;
  std::ifstream gt_file;
  if(!gt_filepath.empty()) {
    have_gt = true;
    gt_file.open(gt_filepath);
  }
  else throw 1;
  // no need - Tsukuba baseline is in cm
  bool using_kitti = false;
  double error_thr = 0.0;
  if(experiment_config.find("tsukuba") != std::string::npos) {
    printf("---------- Using Tsukuba dataset! ------------\n");
    track_name = "00";
    error_thr = 2.0;
  }
  else if(experiment_config.find("bb2") != std::string::npos) {
    printf("---------- Using Bumblebee dataset! ------------\n");
    track_name = "bb";
  }
  // else it is KITTI
  else {
    printf("---------- Using KITTI dataset! ------------\n");
    track_name = imagelistfn.substr(imagelistfn.size()-10,2);
    using_kitti = true;
    error_thr = 8.0;
  }

  int num_of_motions = ((imagelist.size()/2) - 1);
  std::vector<cv::Mat> gt_world_motion, gt_camera_motion;
  core::FormatHelper::Read2FrameMotionFromAccCameraMotion(gt_filepath, num_of_motions,
                                                          gt_world_motion, gt_camera_motion);

  //std::string bad_folder = "/home/kivan/Projects/datasets/KITTI/tracks/bad_patches/";
  //std::string good_folder = "/home/kivan/Projects/datasets/KITTI/tracks/good_patches/";
  //core::Helper::CleanFolder(bad_folder);
  //core::Helper::CleanFolder(good_folder);

  std::vector<size_t> track_index, track_cnt;
  track_index.assign(stereo_tracker->countFeatures(), 0);
  track_cnt.assign(stereo_tracker->countFeatures(), 0);
  size_t all_tracks_cnt = 0;

  // KITTI
  int h_bins = 15;
  int v_bins = 5;
  //int h_bins = 20;
  //int v_bins = 10;
  // tsukuba
  //int h_bins = 15;
  //int v_bins = 11;
  //int h_bins = 9;
  //int v_bins = 7;

  std::vector<std::vector<double>> left_reproj_errors, right_reproj_errors;
  std::vector<std::vector<Eigen::Vector2d>> left_error_vectors, right_error_vectors;
  left_reproj_errors.resize(h_bins * v_bins);
  right_reproj_errors.resize(h_bins * v_bins);
  left_error_vectors.resize(h_bins * v_bins);
  right_error_vectors.resize(h_bins * v_bins);
  //double scale_factor = 1.2;
  double scale_factor = 1.8;
  cv::Mat error_img = cv::Mat::zeros(scale_factor*img_left.rows, scale_factor*img_left.cols, CV_8UC3);
  //int gt_better_cnt = 0;
  int frame_num = 0;
  auto start = std::chrono::system_clock::now();
  //for(unsigned i = start_frame + 2; i < start_frame+60; i+=2) {
  for(unsigned i = start_frame + 2; i < end_frame; i+=2) {
    int motion_idx = i / 2 - 1;
    frame_num++;
    std::cout << "Frame num: " << frame_num << "\n";
    std::cout << "Motion idx: " << motion_idx << " / " << (end_frame/2-2) << "\n";
    cout << source_folder + imagelist[i] << endl;
    cv::swap(img_left_prev, img_left);
    cv::swap(img_right_prev, img_right);
    img_left = cv::imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);

    stereo_tracker->track(img_left, img_right);

    // default prior
    double max_z_dist = 300.0; // for tsukuba in cm
    double min_disp = 0.1;
    double max_disp_diff = 50.0;
    // for KITTI
    if(using_kitti) {
      max_disp_diff = 30.0;
      max_z_dist = 3.0;         // 3m
    }
    // filter tracks with prior knowledge
    FeatureHelper::FilterTracksWithPriorOld(*stereo_tracker, max_disp_diff, min_disp);
    
    // filter outliers with GT
    bool gt_filtering_passed = false;
    if(have_gt) {
      std::vector<std::vector<double>> left_tmp, right_tmp;
      gt_filtering_passed = track::EvalHelper::FilterOutliersWithGroundtruth(*stereo_tracker,
                             cam_params, gt_world_motion[motion_idx], error_thr); // the best thresh - 4.0
    }


    // draw tracks before ransac
    //if(!gt_filtering_passed || debug)
    //  EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
    //                               "all_left", "all_right");

    // run odometry optimization
    visodom::VisualOdometryRansac* viso_ransac = dynamic_cast<visodom::VisualOdometryRansac*>(viso);
    viso_ransac->setLeftPrevImage(img_left_prev);
    // Rt is the motion of world points with respect to the camera
    Rt = viso->getMotion(*stereo_tracker);
    // TODO - add diff check to test for estimation chrashes on KITTI test
    // CheckMotionDiff(Rt, Rt_prev);
    //if(viso->process(libviso_tracks)) {
    if(!Rt.empty()) {
      // is it better to kill or not to kill outliers in tracker?
      std::vector<int> inliers = viso->getTrackerInliers();
      std::vector<int> outliers = viso->getTrackerOutliers();
      std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      //FeatureHelper::FilterOutlierTracks(*stereo_tracker, active_tracks, inliers, 30.0, 0.1);
      //FeatureHelper::FilterRansacOutliers(*stereo_tracker, active_tracks, inliers);
      for(size_t j = 0; j < outliers.size(); j++)
        stereo_tracker->removeTrack(outliers[j]);
      std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      //// draw tracks
      if(debug)
        EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
                                     "ransac_left", "ransac_right");

      //////FeatureHelper::drawStereoRefinerTracks(tracker_basic, tracker, disp_camera_left, disp_camera_right);
      //waitKey(0);

      //// run GT filter after RANSAC
      //bool filter_bad = false;
      ////if(using_kitti)
      ////  filter_bad = true;
      ////double max_reproj_error = 0.6; // for BA experiment
      //double max_reproj_error = 1.0;
      //double max_remove_ratio = 0.5;  // remove max 50%
      //int bad_tracks = track::EvalHelper::CountFilterStoreBadTracks(*stereo_tracker, cam_params, Rt_gt,
      //    good_folder, bad_folder, track_index, track_cnt, all_tracks_cnt, filter_bad, max_reproj_error,
      //    max_remove_ratio);
      //std::cout << "Num of bad tracks = " << bad_tracks << '\n';

      //track::DebugHelper::DebugStereoRefiner(img_left_prev, img_right_prev, img_left, img_right,
      //    *dynamic_cast<StereoTrackerRefiner*>(stereo_tracker), Rt_gt, cam_params);

      //track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, gt_motion[i/2], img_left_prev,
      //    img_right_prev, "left_prev_ransac", true, left_reproj_errors, right_reproj_errors,
      //    h_bins, v_bins, false);
      if(gt_filtering_passed)
        track::EvalHelper::CalculateReprojectionErrors(*stereo_tracker, cam_params, gt_world_motion[motion_idx],
                                                      img_left.rows, img_left.cols,
                                                      left_reproj_errors, right_reproj_errors,
                                                      left_error_vectors, right_error_vectors, h_bins, v_bins);
      //if(frame_num > 20) {
      if(i == (end_frame-2)) {
        std::vector<double> left_means, right_means, left_variances, right_variances;
        std::vector<Eigen::Vector2d> left_vec_means, left_vec_variances;
        //track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors, error_img, true, false);
        track::EvalHelper::CalculateErrorStatistics(left_reproj_errors, left_error_vectors, left_means,
                                                    left_variances, left_vec_means, left_vec_variances);
        track::EvalHelper::DrawErrorStatistics(v_bins, h_bins, left_means, left_variances,
                                               left_vec_means, left_vec_variances, left_reproj_errors,
                                               true, error_img);
        
        //cv::imshow("error_distribution", error_img);
        //cv::waitKey(100);
        track::EvalHelper::SaveErrorStatistics(v_bins, h_bins, left_reproj_errors, left_means,
                                               left_variances, left_vec_means, left_vec_variances,
                                               track_name + "_statistics_matrix.yml");
        cv::Mat eq_img;
        //cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
        //cv::equalizeHist(eq_img, eq_img);
        //cv::imwrite("heat_map_with_stats.png", eq_img);
        cv::imwrite(track_name + "_heat_map_with_stats.png", error_img);
        track::EvalHelper::DrawErrorStatistics(v_bins, h_bins, left_means, left_variances,
                                               left_vec_means, left_vec_variances, left_reproj_errors,
                                               false, error_img);
        cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
        cv::equalizeHist(eq_img, eq_img);
        cv::imwrite(track_name + "_heat_map_orig.png", error_img);
        cv::imwrite(track_name + "_heat_map.png", eq_img);
      }

      if(have_gt)
        std::cout << "groundtruth:\n" << gt_world_motion[motion_idx] << "\n";
      std::cout << "two-frame odometry:\n" << Rt << "\n";
      //if(!gt_filtering_passed || debug)
      //  while(cv::waitKey(0) != 27);

      //track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, Rt, img_left_prev, img_right_prev);
      // TODO:
      //Eigen::Vector3d t, r;
      //core::MathHelper::GetEulerAngles(Rt, r);
      //core::MathHelper::GetTranslation(Rt, t);

      // if the motion is to small skip it
      // tracker.undo_state();
      core::MathHelper::invTrans(Rt, Rt_inv);    // better
      extr_params.push_back(Rt_inv.clone());
      // on success, update current pose
      pose = pose * Rt_inv;
      libviso_Rt_all.push_back(pose.clone());

      //double num_matches = viso->getNumberOfMatches();
      //double num_inliers = viso->getNumberOfInliers();
      //cout << "[Libviso] Matches: " << num_matches << ", Inliers: " << num_inliers << " -> "
      //     << 100.0*num_inliers/num_matches << " %" << endl;
      std::cout << "-----------------------------------------------------------------------\n\n";
      cv::Mat location_viso(pose_libviso, cv::Range(0,4), cv::Range(3,4)); // extract 4-th column
      location_viso.copyTo(prev_location_viso);
    } else {
      std::cout << "libviso ... failed!" << endl;
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

  std::cout << "Writing track " << track_name << " to files\n";
  std::ofstream libviso_ofile(output_folder + '/' + track_name + ".txt");

  for(size_t i = 0; i < libviso_Rt_all.size(); i++) {
    FormatHelper::WriteMatRt(libviso_Rt_all[i], libviso_ofile);
  }

  delete detector;
  delete mono_tracker;
  delete stereo_tracker;
  delete viso;

  //// run evaluation
  //std::string cmd = "/home/kivan/Projects/cv-stereo/stereo_odometry/evaluation/evaluate_odometry_dense 7 7 " +
  //    std::to_string(start_frame/2) + " " + std::to_string(nframes) + " ./results/ eval_stats";
  //std::cout << cmd << "\n";
  //system(cmd.c_str());
}


int main(int argc, char** argv)
{
  std::string config_file;
  std::string experiment_config;
  std::string imagelistfn;
  std::string cam_params_file;
  std::string source_folder;
  std::string gt_filepath;

  if(argc != 5){
    std::cout << "Usage:\n" << argv[0] << " -c dataset_config -e experiment_config\n";
    return 0;
  }

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help, h", "produce help message")
      ("config,c", po::value<string>(&config_file), "config filename")
      ("experiment_config,e", po::value<string>(&experiment_config), "experiment config filename");
    po::options_description cmdline_options;
    cmdline_options.add(generic);

    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
    if(vm.count("help")) {
      cout << generic;
      return 0;
    }

    po::options_description config("Config file options");
    config.add_options()
      ("camera_params,p", po::value<std::string>(&cam_params_file)->default_value("camera_params.txt"), "camera params file")
      ("source_folder,s", po::value<std::string>(&source_folder), "folder with source")
      ("imglist,l", po::value<std::string>(&imagelistfn), "file with image list")
      ("groundtruth,g", po::value<std::string>(&gt_filepath), "file with motion GT");
    
    std::ifstream ifs(config_file);
    if(!ifs)
      throw "can not open config file: " + config_file + "\n";
    else {
      po::store(parse_config_file(ifs, config, true), vm);
      notify(vm);
    }
  }
  catch(std::exception& e) {
    cout << e.what() << "\n";
    return -1;
  }

  std::cout << "Using track config = " << config_file << '\n';
  run_visual_odometry(source_folder, imagelistfn, experiment_config, cam_params_file, gt_filepath);

  return 0;
}
