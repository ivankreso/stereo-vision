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

  int mf_ba_frames = 3;
  optim::BundleAdjusterBase* bundle_adjuster_multiframe = new optim::BundleAdjusterMultiframe(
      mf_ba_frames-1, stereo_tracker->countFeatures(), optim::SBAbase::kMotion, true);
  if(use_ba) {
    bundle_adjuster->set_camera_params(cam_params);
    bundle_adjuster_multiframe->set_camera_params(cam_params);
  }

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
  std::vector<std::vector<double>> left_reproj_errors, right_reproj_errors;
  left_reproj_errors.resize(h_bins * v_bins);
  right_reproj_errors.resize(h_bins * v_bins);
  double scale_factor = 1.2;
  cv::Mat error_img = cv::Mat::zeros(scale_factor*img_left.rows, scale_factor*img_left.cols, CV_8UC3);

  //int gt_better_cnt = 0;
  int image_num;
  int frame_num = 0;
  auto start = std::chrono::system_clock::now();
  //for(unsigned i = start_frame + 2; i < start_frame+60; i+=2) {
  for(unsigned i = start_frame + 2; i < end_frame; i+=2) {
    image_num = i / 2;
    frame_num++;
    std::cout << "Frame num: " << frame_num << "\n";
    std::cout << "Image num: " << image_num << " / " << end_frame/2 << "\n";
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

    // reading the GT trans
    //if(have_gt) {
    //  FormatHelper::ReadNextRtMatrix(gt_file, Rt_gt_curr);
    //  cv::Mat Rt_gt_inv = Rt_gt_prev * Rt_gt_curr;
    //  MathHelper::invTrans(Rt_gt_inv, Rt_gt);
    //  MathHelper::invTrans(Rt_gt_curr, Rt_gt_prev);
    //  //cout << Rt_gt_inv << endl;

    //  // run GT filter before libviso
    //  //bool filter_bad = false;
    //  //if(using_kitti)
    //  //  filter_bad = true;
    //  //double max_reproj_error = 0.6;
    //  //double max_remove_ratio = 0.5;  // remove max 50%
    //  //bool store_patches = false;
    //  //int bad_tracks = track::EvalHelper::CountFilterStoreBadTracks(*stereo_tracker, cam_params, Rt_gt,
    //  //    store_patches, good_folder, bad_folder, track_index, track_cnt, all_tracks_cnt, filter_bad,
    //  //    max_reproj_error, max_remove_ratio);
    //  //std::cout << "Num of bad tracks = " << bad_tracks << '\n';
    //}

    // filter outliers with GT
    //if(have_gt) {
    //  std::vector<std::vector<double>> left_tmp, right_tmp;
    //  track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, gt_motion[i/2], img_left_prev,
    //      img_right_prev, "left_prev_best", false, left_tmp, right_tmp,
    //      h_bins, v_bins, false, 4.0, true); // the best - 4.0
    //}

    // convert from my tracker to Libviso data structure
    //std::vector<int> active_tracks;
    //std::vector<libviso::Matcher::p_match> libviso_tracks;
    //FeatureHelper::TrackerBaseToLibviso(stereo_tracker, libviso_tracks, active_tracks);

    // draw tracks before ransac
    if(debug)
      EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
                                   "all_left", "all_right");

    // run odometry optimization
    visodom::VisualOdometryRansac* viso_ransac = dynamic_cast<visodom::VisualOdometryRansac*>(viso);
    viso_ransac->setLeftPrevImage(img_left_prev);
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
      //if(frame_num > 10)
      //  track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors, error_img, true, false);
      //if(i == (end_frame-2)) {
      ////if(i == 2*20) {
      //  cv::Mat eq_img;
      //  cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
      //  cv::equalizeHist(eq_img, eq_img);
      //  cv::imwrite("heat_map_with_stats_orig.png", error_img);
      //  cv::imwrite("heat_map_with_stats.png", eq_img);
      //  track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors, error_img, false, true);
      //  cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
      //  cv::equalizeHist(eq_img, eq_img);
      //  cv::imwrite("heat_map_orig.png", error_img);
      //  cv::imwrite("heat_map.png", eq_img);
      //}
      
      //libviso::Matrix Rt_libviso = viso->getMotion();
      //MathHelper::matrixToMat(Rt_libviso, Rt);
      if(have_gt)
        cout << "groundtruth:\n" << gt_motion[i/2] << endl;
      cout << "two-frame odometry:\n" << Rt << endl;
      if(debug)
        while(cv::waitKey(0) != 27);

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

      if(use_ba) {
        // ----- SBA -----
        //bundle_adjuster->update_tracks(*stereo_tracker, Rt_inv); TODO - old BA
        bundle_adjuster->update_tracks(*stereo_tracker, Rt);
        bundle_adjuster_multiframe->update_tracks(*stereo_tracker, Rt_inv);
        //sba->updateTracks(img_left, img_right, *stereo_tracker, mat_I, cam_params);
        //if(frame_num >= ba_window_size) {
        if(frame_num >= 2) {
          double libviso_error = EvalHelper::GetStereoReprojError(*stereo_tracker, cam_params, Rt);
          double gt_error;
          if(have_gt)
            gt_error = EvalHelper::GetStereoReprojError(*stereo_tracker, cam_params, gt_motion[i/2]);
          bundle_adjuster->optimize();
          cv::Mat Rt_sba = bundle_adjuster->camera_motion(bundle_adjuster->camera_motion_num() - 1);
          cv::Mat Rt_sba_inv;
          MathHelper::invTrans(Rt_sba, Rt_sba_inv);
          if(use_ba)
            std::cout << "two-frame BA odometry:\n" << Rt_sba_inv << endl;

          double ba_error = EvalHelper::GetStereoReprojError(*stereo_tracker, cam_params, Rt_sba_inv);
          if(have_gt) printf("GT reproj error = %e\n", gt_error);
          printf("Libviso reproj error = %e\n", libviso_error);
          printf("BA reproj error = %e\n", ba_error);
        }
        if(frame_num >= (mf_ba_frames-1)) {
          bundle_adjuster_multiframe->optimize();
          cv::Mat Rt_sba = bundle_adjuster_multiframe->camera_motion(
                        bundle_adjuster_multiframe->camera_motion_num() - 1);
          cv::Mat Rt_sba_inv;
          MathHelper::invTrans(Rt_sba, Rt_sba_inv);
          cout << "multi-frame BA odometry:\n" << Rt_sba_inv << endl;
        }
      }

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

  std::ofstream ba_ofile;
  std::ofstream ba_ofile2;
  //cv::Mat sba_pose;
  if(use_ba) {
    std::string ba_folder = output_folder.substr(0, output_folder.size()-1) + "_ba/";
    std::string ba_folder2 = output_folder.substr(0, output_folder.size()-1) + "_ba_multiframe/";
    ba_ofile.open(ba_folder + track_name + ".txt");
    ba_ofile2.open(ba_folder2 + track_name + ".txt");
    //sba_pose = cv::Mat::eye(4, 4, CV_64F);
  }
  for(size_t i = 0; i < libviso_Rt_all.size(); i++) {
    FormatHelper::WriteMatRt(libviso_Rt_all[i], libviso_ofile);
    if(use_ba)
      FormatHelper::WriteMatRt(bundle_adjuster->camera_motion_acc(i), ba_ofile);

    FormatHelper::WriteMatRt(bundle_adjuster_multiframe->camera_motion_acc(i), ba_ofile2);
  }

  if(use_ba)
    delete bundle_adjuster;
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
