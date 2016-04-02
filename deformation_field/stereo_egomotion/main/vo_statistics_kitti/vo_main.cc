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

int img_rows_;
int img_cols_;

void run_visual_odometry(const std::string& source_folder,
                         const std::string& imagelistfn,
                         const std::string& experiment_config,
                         const std::string& cparams_file,
                         const std::string& gt_filepath,
                         std::vector<std::vector<double>>& left_reproj_errors,
                         std::vector<std::vector<double>>& right_reproj_errors,
                         std::vector<std::vector<Eigen::Vector2d>>& left_error_vectors,
                         std::vector<std::vector<Eigen::Vector2d>>& right_error_vectors)
{
  std::deque<cv::Mat> extr_params; // pose mat
  std::deque<cv::Mat> Rt_params;

  double cam_params[5];
  FormatHelper::readCameraParams(cparams_file, cam_params);

  std::string output_folder;
  FeatureDetectorBase* detector = nullptr;
  TrackerBase* mono_tracker = nullptr;
  StereoTrackerBase* stereo_tracker = nullptr;
  visodom::VisualOdometryBase* viso = nullptr;
  //optim::FeatureHelperSBA* bundle_adjuster = nullptr;
  optim::BundleAdjusterBase* bundle_adjuster = nullptr;
  bool use_ba; // use bundle adjustment
  track::ExperimentFactory::create_experiment(experiment_config, cam_params, output_folder, &detector,
                                              &mono_tracker, &stereo_tracker, &viso, use_ba, &bundle_adjuster);

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

  bool smooth_images = false;
  cv::Mat img_left_prev, img_right_prev;
  std::cout << source_folder + imagelist[start_frame] << "\n";
  cv::Mat img_left = cv::imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  if(img_left.empty()) {
    std::cout << "Error: no images!\n";
    throw 1;
  }
  if(smooth_images) {
    cv::GaussianBlur(img_left, img_left, cv::Size(3,3), 0.7);
    cv::GaussianBlur(img_right, img_right, cv::Size(3,3), 0.7);
  }

  //libviso::Matrix pose = libviso::Matrix::eye(4);
  cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  //Matrix point_rt = Matrix::eye(4);
  //cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat Rt_inv;
  cv::Mat mat_I = cv::Mat::eye(4, 4, CV_64F);
  mat_I.copyTo(Rt_inv);
  Vec<double,7> trans_vec;
  extr_params.push_back(mat_I.clone());
  cv::Mat Rt(4, 4, CV_64F);
  cv::Mat Rt_gt(4, 4, CV_64F);
  cv::Mat Rt_gt_prev = cv::Mat::eye(4, 4, CV_64F);
  cv::Mat Rt_gt_curr = cv::Mat::eye(4, 4, CV_64F);
  MathHelper::invTrans(Rt_inv, Rt);
  Rt_params.push_back(Rt.clone());

  cv::Mat prev_location_viso = cv::Mat::zeros(4, 1, CV_64F);

  // init the tracker
  stereo_tracker->init(img_left, img_right);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<cv::Mat> libviso_Rt_all;
  cv::Mat pose_libviso = Mat::eye(4, 4, CV_64F);
  libviso_Rt_all.push_back(pose_libviso.clone());

  std::string track_name;
  bool have_gt = false;
  std::ifstream gt_file;
  if(!gt_filepath.empty()) {
    have_gt = true;
    gt_file.open(gt_filepath);
  }
  // no need - Tsukuba baseline is in cm
  bool using_kitti = false;
  if(experiment_config.find("tsukuba") != std::string::npos) {
    printf("---------- Using Tsukuba dataset! ------------\n");
    track_name = "00";
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
  }

  std::vector<cv::Mat> gt_motion;
  if(have_gt) {
    // read all GT data
    // skip first (identety) matrix
    FormatHelper::ReadNextRtMatrix(gt_file, Rt_gt_prev);
    MathHelper::invTrans(Rt_gt_prev, Rt_gt);
    gt_motion.push_back(Rt_gt.clone());
    int num_motions = ((imagelist.size()/2) - 1);
    for(int i = 0; i < num_motions; i++) {
      FormatHelper::ReadNextRtMatrix(gt_file, Rt_gt_curr);
      cv::Mat Rt_gt_inv = Rt_gt_prev * Rt_gt_curr;
      MathHelper::invTrans(Rt_gt_inv, Rt_gt);
      MathHelper::invTrans(Rt_gt_curr, Rt_gt_prev);
      gt_motion.push_back(Rt_gt.clone());
    }
    Rt_gt.release();
  }

  std::string bad_folder = "/home/kivan/Projects/datasets/KITTI/tracks/bad_patches/";
  std::string good_folder = "/home/kivan/Projects/datasets/KITTI/tracks/good_patches/";
  core::Helper::CleanFolder(bad_folder);
  core::Helper::CleanFolder(good_folder);

  std::vector<size_t> track_index, track_cnt;
  track_index.assign(stereo_tracker->countFeatures(), 0);
  track_cnt.assign(stereo_tracker->countFeatures(), 0);
  size_t all_tracks_cnt = 0;

  //int h_bins = 20;
  //int v_bins = 10;
  int h_bins = 15;
  int v_bins = 5;
  left_reproj_errors.clear();
  right_reproj_errors.clear();
  left_error_vectors.clear();
  right_error_vectors.clear();
  left_reproj_errors.resize(h_bins * v_bins);
  right_reproj_errors.resize(h_bins * v_bins);
  left_error_vectors.resize(h_bins * v_bins);
  right_error_vectors.resize(h_bins * v_bins);
  //double scale_factor = 1.2;
  double scale_factor = 1.8;
  cv::Mat error_img = cv::Mat::zeros(scale_factor*img_left.rows, scale_factor*img_left.cols, CV_8UC3);
  img_rows_ = img_left.rows;
  img_cols_ = img_left.cols;
  //int gt_better_cnt = 0;
  int image_num;
  int frame_num = 0;
  auto start = std::chrono::system_clock::now();
  //for(unsigned i = start_frame + 2; i < start_frame+60; i+=2) {
  for(unsigned i = start_frame + 2; i < end_frame; i+=2) {
    image_num = i / 2;
    frame_num++;
    std::cout << "Frame num: " << frame_num << "\n";
    std::cout << "Sequance image num: " << image_num << " / " << (end_frame/2-1) << "\n";
    cout << source_folder + imagelist[i] << endl;
    cv::swap(img_left_prev, img_left);
    cv::swap(img_right_prev, img_right);
    img_left = cv::imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    if(smooth_images) {
      cv::GaussianBlur(img_left, img_left, cv::Size(3,3), 0.7);
      cv::GaussianBlur(img_right, img_right, cv::Size(3,3), 0.7);
    }

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
                             cam_params, gt_motion[i/2], 4.0); // the best thresh - 4.0
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
        track::EvalHelper::CalcuateReprojectionErrors(*stereo_tracker, cam_params, gt_motion[i/2],
                                                      img_left.rows, img_left.cols,
                                                      left_reproj_errors, right_reproj_errors,
                                                      left_error_vectors, right_error_vectors, h_bins, v_bins);
      //if(frame_num > 2) {
      if(i == (end_frame-2)) {
        std::vector<double> means, variances;
        std::vector<Eigen::Vector2d> vec_means, vec_variances;
        //track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors, error_img, true, false);
        track::EvalHelper::CalculateErrorStatistics(left_reproj_errors, right_reproj_errors,
                                                    left_error_vectors, right_error_vectors,
                                                    means, variances, vec_means, vec_variances);
        track::EvalHelper::DrawErrorStatistics(v_bins, h_bins, means, variances,
                                               vec_means, vec_variances, left_reproj_errors,
                                               right_reproj_errors, true, error_img);
        
        //cv::imshow("error_distribution", error_img);
        //cv::waitKey(100);
        track::EvalHelper::SaveErrorStatistics(v_bins, h_bins, means, variances,
                                               vec_means, vec_variances, track_name + "_statistics_matrix.yml");
        cv::Mat eq_img;
        //cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
        //cv::equalizeHist(eq_img, eq_img);
        //cv::imwrite("heat_map_with_stats.png", eq_img);
        cv::imwrite(track_name + "_heat_map_with_stats.png", error_img);
        track::EvalHelper::DrawErrorStatistics(v_bins, h_bins, means, variances,
                                               vec_means, vec_variances, left_reproj_errors,
                                               right_reproj_errors, false, error_img);
        cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
        cv::equalizeHist(eq_img, eq_img);
        cv::imwrite(track_name + "_heat_map_orig.png", error_img);
        cv::imwrite(track_name + "_heat_map.png", eq_img);
      }

      if(have_gt)
        cout << "groundtruth:\n" << gt_motion[i/2] << endl;
      cout << "two-frame odometry:\n" << Rt << endl;
      //if(!gt_filtering_passed || debug)
      //  while(cv::waitKey(0) != 27);

      //track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, Rt, img_left_prev, img_right_prev);
      // TODO:
      //Eigen::Vector3d t, r;
      //core::MathHelper::GetEulerAngles(Rt, r);
      //core::MathHelper::GetTranslation(Rt, t);

      // if the motion is to small skip it
      // tracker.undo_state();
      MathHelper::invTrans(Rt, Rt_inv);    // better
      extr_params.push_back(Rt_inv.clone());
      // on success, update current pose
      pose = pose * Rt_inv;
      libviso_Rt_all.push_back(pose.clone());

      //double num_matches = viso->getNumberOfMatches();
      //double num_inliers = viso->getNumberOfInliers();
      //cout << "[Libviso] Matches: " << num_matches << ", Inliers: " << num_inliers << " -> "
      //     << 100.0*num_inliers/num_matches << " %" << endl;
      cout << "-----------------------------------------------------------------------\n\n";
      Mat location_viso(pose_libviso, Range(0,4), Range(3,4)); // extract 4-th column
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


void run_kitti(std::string config_file, std::string experiment_config,
               std::vector<std::vector<double>>& left_reproj_errors,
               std::vector<std::vector<double>>& right_reproj_errors,
               std::vector<std::vector<Eigen::Vector2d>>& left_error_vectors,
               std::vector<std::vector<Eigen::Vector2d>>& right_error_vectors)
{
  std::string imagelistfn;
  std::string cam_params_file;
  std::string source_folder;
  std::string gt_filepath;

  try {
    po::variables_map vm;    
    po::options_description config("Config file options");
    config.add_options()
      ("camera_params,p", po::value<std::string>(&cam_params_file)->default_value("camera_params.txt"), "camera params file")
      ("source_folder,s", po::value<std::string>(&source_folder), "folder with source")
      ("imglist,l", po::value<std::string>(&imagelistfn), "file with image list")
      ("groundtruth,g", po::value<std::string>(&gt_filepath), "file with motion GT");
    
    std::ifstream ifs(config_file);
    if(!ifs) {
      std::cerr << "can not open config file: " + config_file + "\n";
      throw 1;
    }
    else {
      po::store(parse_config_file(ifs, config, true), vm);
      notify(vm);
    }
  }
  catch(std::exception& e) {
    cout << e.what() << "\n";
    throw -1;
  }

  std::cout << "Using track config = " << config_file << '\n';
  run_visual_odometry(source_folder, imagelistfn, experiment_config, cam_params_file, gt_filepath,
                      left_reproj_errors, right_reproj_errors, left_error_vectors, right_error_vectors);
}

int main(int argc, char** argv)
{
  std::vector<std::vector<double>> left_reproj_errors, right_reproj_errors;
  std::vector<std::vector<double>> left_reproj_errors_all, right_reproj_errors_all;
  std::vector<std::vector<Eigen::Vector2d>> left_error_vectors, right_error_vectors;
  std::vector<std::vector<Eigen::Vector2d>> left_error_vectors_all, right_error_vectors_all;

  for(int i = 0; i < 11; i++) {
    std::stringstream track_num;
    track_num << setfill('0') << setw(2) << i;
    run_kitti("../../../config_files/config_kitti_" + track_num.str() + ".txt",
              "../../../config_files/experiments/kitti/tracker_ncc_test.txt",
              left_reproj_errors, right_reproj_errors, left_error_vectors, right_error_vectors);
    left_reproj_errors_all.insert(left_reproj_errors_all.begin(), left_reproj_errors.begin(), left_reproj_errors.end());
    right_reproj_errors_all.insert(right_reproj_errors_all.begin(), right_reproj_errors.begin(), right_reproj_errors.end());
    left_error_vectors_all.insert(left_error_vectors_all.begin(), left_error_vectors.begin(), left_error_vectors.end());
    right_error_vectors_all.insert(right_error_vectors_all.begin(), right_error_vectors.begin(), right_error_vectors.end());
  }

  int h_bins = 15;
  int v_bins = 5;
  //double scale_factor = 1.2;
  double scale_factor = 1.8;
  cv::Mat error_img = cv::Mat::zeros(scale_factor*img_rows_, scale_factor*img_cols_, CV_8UC3);

  std::vector<double> means, variances;
  std::vector<Eigen::Vector2d> vec_means, vec_variances;
  //track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors, error_img, true, false);
  track::EvalHelper::CalculateErrorStatistics(left_reproj_errors_all, right_reproj_errors_all,
                                              left_error_vectors_all, right_error_vectors_all,
                                              means, variances, vec_means, vec_variances);
  track::EvalHelper::DrawErrorStatistics(v_bins, h_bins, means, variances,
                                         vec_means, vec_variances, left_reproj_errors_all,
                                         right_reproj_errors_all, true, error_img);

  //cv::imshow("error_distribution", error_img);
  //cv::waitKey(100);
  track::EvalHelper::SaveErrorStatistics(v_bins, h_bins, means, variances,
                                         vec_means, vec_variances, "kitti_statistics_matrix.yml");
  cv::Mat eq_img;
  //cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
  //cv::equalizeHist(eq_img, eq_img);
  //cv::imwrite("heat_map_with_stats.png", eq_img);
  cv::imwrite("kitti_heat_map_with_stats.png", error_img);
  track::EvalHelper::DrawErrorStatistics(v_bins, h_bins, means, variances,
                                         vec_means, vec_variances, left_reproj_errors,
                                         right_reproj_errors, false, error_img);
  cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
  cv::equalizeHist(eq_img, eq_img);
  cv::imwrite("kitti_heat_map_orig.png", error_img);
  cv::imwrite("kitti_heat_map.png", eq_img);

  return 0;
}
