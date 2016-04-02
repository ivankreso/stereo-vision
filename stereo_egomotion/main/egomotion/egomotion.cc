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
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../core/image.h"
#include "../../../core/helper.h"
#include "../../../core/math_helper.h"
#include "../../../stereo_egomotion/base/egomotion_base.h"
#include "../../../tracker/stereo/experiment_factory.h"
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
//#include "../../../tracker/refiner/feature_refiner_klt.h"
#include "../../../tracker/detector/feature_detector_base.h"
#include "../../../tracker/base/tracker_helper.h"
#include "../../../optimization/bundle_adjustment/bundle_adjuster_base.h"
//#include "../../../optimization/bundle_adjustment/bundle_adjuster_multiframe.h"
#include "../../math_helper.h"
#include "../../../tracker/base/eval_helper.h"
#include "../../../core/format_helper.h"
//#include "../../helper_libviso.h"
//#include "../../cv_plotter.h"

using namespace core;

void RunEgomotionExperiment(const std::string& source_folder,
                            const std::string& imagelistfn,
                            const std::string& config_filepath,
                            const std::string& experiment_config,
                            const std::string& cparams_file,
                            const std::string& gt_filepath,
                            const std::string& deformation_field_path,
                            std::string& output_folder)
{
  std::string track_name;
  bool have_gt = false;
  std::ifstream gt_file;
  if (!gt_filepath.empty()) {
    have_gt = true;
    gt_file.open(gt_filepath);
  }
  // no need - Tsukuba baseline is in cm
  bool using_kitti = false;
  if (experiment_config.find("tsukuba") != std::string::npos) {
    printf("---------- Using Tsukuba dataset! ------------\n");
    track_name = "00";
  }
  else if (experiment_config.find("bb2") != std::string::npos) {
    printf("---------- Using Bumblebee dataset! ------------\n");
    track_name = "bb";
  }
  // else it is KITTI
  else {
    printf("---------- Using KITTI dataset! ------------\n");
    track_name = imagelistfn.substr(imagelistfn.size()-10,2);
    using_kitti = true;
  }

  bool serialize_tracker = false;
  bool use_serialized_data = false;
  const std::string serialize_folder = "/mnt/ssd/kivan/datasets/serialized_tracker/"
      + track_name + "/";
  if (serialize_tracker) {
    struct stat st = {0};
    if(stat(serialize_folder.c_str(), &st) == -1)
      mkdir(serialize_folder.c_str(), 0700);
  }

  double cam_params[5];
  FormatHelper::readCameraParams(cparams_file, cam_params);
  Eigen::VectorXd camera_params(5);
  for (int i = 0; i < 5; i++)
    camera_params[i] = cam_params[i];

  std::vector<std::string> imagelist;
  bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
  if (!ok || imagelist.empty())
    throw "can not open " + imagelistfn + " or the string list is empty\n";

  size_t start_frame = 0 * 2;
  //size_t start_frame = 770 * 2;
  size_t end_frame = imagelist.size();
  //int nframes = (imagelist.size() - start_frame) / 2;
  bool debug = false;
  //debug = true;

  std::cout << source_folder + imagelist[start_frame] << "\n";
  cv::Mat img_left_prev, img_right_prev;
  cv::Mat img_left = cv::imread(source_folder+imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder+imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  if (img_left.empty()) {
    std::cout << "Error: no images!\n";
    throw 1;
  }
  int img_rows = img_left.rows;
  int img_cols = img_left.cols;
  bool use_ba; // use bundle adjustment
  track::FeatureDetectorBase* detector = nullptr;
  track::TrackerBase* mono_tracker = nullptr;
  track::StereoTrackerBase* stereo_tracker = nullptr;
  egomotion::EgomotionBase* egomotion = nullptr;
  optim::BundleAdjusterBase* bundle_adjuster = nullptr;
  bool status = track::ExperimentFactory::create_experiment(
      experiment_config, deformation_field_path, cam_params, img_rows, img_cols, output_folder,
      &detector, &mono_tracker, &stereo_tracker, &egomotion, use_ba, &bundle_adjuster);
  if (!status) {
    std::cout << "Invalid experiment config!\n";
    return;
  }

  //int mf_ba_frames = 3;
  //optim::BundleAdjusterBase* bundle_adjuster_multiframe = new optim::BundleAdjusterMultiframe(
  //    mf_ba_frames-1, stereo_tracker->countFeatures(), optim::SBAbase::kMotion, true);
  if (use_ba) {
    bundle_adjuster->SetCameraParams(camera_params);
    //bundle_adjuster_multiframe->set_camera_params(cam_params);
  }

  //egomotion::Matrix pose = egomotion::Matrix::eye(4);
  //cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Rt = Eigen::Matrix4d::Identity();
  //Eigen::Matrix4d Rt_inv = Eigen::Matrix4d::Identity();
  //Matrix point_rt = Matrix::eye(4);
  //cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);
  //cv::Mat Rt_inv;
  //cv::Mat mat_I = cv::Mat::eye(4, 4, CV_64F);
  //mat_I.copyTo(Rt_inv);
  //cv::Mat Rt(4, 4, CV_64F);
  //cv::Mat Rt_prev(4, 4, CV_64F);
  //cv::Mat Rt_gt(4, 4, CV_64F);
  //cv::Mat Rt_gt_prev = cv::Mat::eye(4, 4, CV_64F);
  //cv::Mat Rt_gt_curr = cv::Mat::eye(4, 4, CV_64F);
  //core::MathHelper::invTrans(Rt_inv, Rt);

  // init the tracker
  if (!use_serialized_data)
    stereo_tracker->init(img_left, img_right);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<Eigen::Matrix4d> egomotion_Rt_all;
  //Eigen::Matrix4d pose_egomotion = Eigen::Matrix4d::Identity();
  egomotion_Rt_all.push_back(pose);

  std::vector<cv::Mat> gt_world_motion, gt_camera_motion;
  if (have_gt) {
    int num_of_motions = ((imagelist.size()/2) - 1);
    core::FormatHelper::Read2FrameMotionFromAccCameraMotion(gt_filepath, num_of_motions,
                                                            gt_world_motion, gt_camera_motion);
  }

  //std::string bad_folder = "/home/kivan/Projects/datasets/KITTI/tracks/bad_patches/";
  //std::string good_folder = "/home/kivan/Projects/datasets/KITTI/tracks/good_patches/";
  //core::Helper::CleanFolder(bad_folder);
  //core::Helper::CleanFolder(good_folder);

  std::vector<size_t> track_index, track_cnt;
  track_index.assign(stereo_tracker->countFeatures(), 0);
  track_cnt.assign(stereo_tracker->countFeatures(), 0);

  int frame_num = 0;
  auto start = std::chrono::system_clock::now();
  //for (unsigned i = start_frame + 2; i < start_frame+60; i+=2) {
  //double max_trans_error = 0.0;
  for (size_t i = start_frame + 2; i < end_frame; i+=2) {
    size_t motion_idx = i / 2 - 1;
    frame_num++;
    std::cout << "Frame num: " << frame_num << " / " << end_frame/2 << "\n";
    if (have_gt)
      std::cout << "motion_idx: " << motion_idx << " / " << (gt_world_motion.size() - 1) << "\n";
    std::cout << source_folder + imagelist[i] << "\n";

    if (!use_serialized_data) {
      cv::swap(img_left_prev, img_left);
      cv::swap(img_right_prev, img_right);
      img_left = cv::imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
      img_right = cv::imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
      stereo_tracker->track(img_left, img_right);
    }
    // serialization code
    else if (!serialize_tracker) {
      std::string serialize_path = serialize_folder + "/tracks_" + std::to_string(motion_idx);
      std::cout << "Loading serialized data at: " << serialize_path << "\n";
      std::ifstream input_file(serialize_path);
      boost::archive::binary_iarchive iarchive(input_file);
      iarchive >> *(dynamic_cast<track::StereoTracker*>(stereo_tracker));
      //track::FeatureInfo data = stereo_tracker->featureLeft(54);
      //std::cout << "Age = " << data.age_ << "\n";
    }
    else return;
    if (serialize_tracker) {
      std::cout << "Serializing tracker data to disk...\n";
      // save data to archive
      std::ofstream ofs(serialize_folder + "/tracks_" + std::to_string(motion_idx));
      //boost::archive::text_oarchive oarchive(ofs);
      boost::archive::binary_oarchive oarchive(ofs);
      // write class instance to archive
      oarchive << *(dynamic_cast<track::StereoTracker*>(stereo_tracker));
      // archive and stream closed when destructors are called
    }

    // default prior
    //double max_z_dist = 300.0; // for tsukuba in cm
    double min_disp = 0.1;
    double max_disp_diff = 50.0;
    // for KITTI
    if (using_kitti) {
      max_disp_diff = 30.0;
      //max_z_dist = 3.0;         // 3m
    }
    // filter tracks with prior knowledge
    // TODO
    track::TrackerHelper::FilterTracksWithPriorOld(*stereo_tracker, max_disp_diff, min_disp);

    //// filter outliers with GT
    //bool gt_filtering_passed = false;
    //if(have_gt) {
    //  std::vector<std::vector<double>> left_tmp, right_tmp;
    //  gt_filtering_passed = track::EvalHelper::FilterOutliersWithGroundtruth(*stereo_tracker,
    //                         cam_params, gt_motion[i/2], 4.0); // the best thresh - 4.0
    //}

    // filter outliers with GT
    //if (have_gt) {
    //  std::vector<std::vector<double>> left_tmp, right_tmp;
    //  track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, gt_motion[i/2],
    //                                         img_left_prev,
    //      img_right_prev, "left_prev_best", false, left_tmp, right_tmp,
    //      h_bins, v_bins, false, 4.0, true); // the best - 4.0
    //}

    // convert from my tracker to egomotion data structure
    //std::vector<int> active_tracks;
    //std::vector<egomotion::Matcher::p_match> egomotion_tracks;
    //FeatureHelper::TrackerBaseToegomotion(stereo_tracker, egomotion_tracks, active_tracks);

    // draw tracks before ransac
    if (debug)
      track::EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
                                   "all_left", "all_right");

    // run egomotion optimization
    // Rt is the motion of world points with respect to the camera
    bool success = egomotion->GetMotion(*stereo_tracker, Rt);
    //bool success = true;

    // diff check to test for estimation chrashes on KITTI test
    //double trans_diff;
    //core::MathHelper::GetMotionError(Rt_prev, Rt, trans_diff);
    //  std::cout << "Trans diff = " << trans_diff << "\n";
    //if (frame_num > 1 && trans_diff > 0.1) {
    //  std::cout << "[Error]: Big trans diff = " << trans_diff << "\n";
    //  throw 1;
    //}

    // CheckMotionDiff(Rt, Rt_prev);
    //if (egomotion->process(egomotion_tracks)) {
    if (success) {
      // is it better to kill or not to kill outliers in tracker?

      // filter outlier tracks
      std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      std::vector<int> outliers = egomotion->GetTrackerOutliers();
      for (size_t j = 0; j < outliers.size(); j++)
        stereo_tracker->removeTrack(outliers[j]);
      std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      ////track::FeatureInfo data = stereo_tracker->featureLeft(54);
      ////std::cout << "After filtering Age = " << data.age_ << "\n";

      //// draw tracks
      if (debug)
        track::EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
                                            "ransac_left", "ransac_right");

      //////FeatureHelper::drawStereoRefinerTracks(tracker_basic, tracker, disp_camera_left,
      //                                           disp_camera_right);
      //waitKey(0);

      //// run GT filter after RANSAC
      //bool filter_bad = false;
      ////if (using_kitti)
      ////  filter_bad = true;
      ////double max_reproj_error = 0.6; // for BA experiment
      //double max_reproj_error = 1.0;
      //double max_remove_ratio = 0.5;  // remove max 50%
      //int bad_tracks = track::EvalHelper::
      //    CountFilterStoreBadTracks(*stereo_tracker, cam_params, Rt_gt,
      //    good_folder, bad_folder, track_index, track_cnt, all_tracks_cnt,
      //    filter_bad, max_reproj_error, max_remove_ratio);
      //std::cout << "Num of bad tracks = " << bad_tracks << '\n';

      //track::DebugHelper::DebugStereoRefiner(img_left_prev, img_right_prev, img_left, img_right,
      //    *dynamic_cast<StereoTrackerRefiner*>(stereo_tracker), Rt_gt, cam_params);

      //track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, gt_motion[i/2],
      //                                       img_left_prev, img_right_prev, "left_prev_ransac",
      //                                       true, left_reproj_errors, right_reproj_errors,
      //                                       h_bins, v_bins, false);
      //if (frame_num > 10)
      //  track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors, error_img,
      //                                           true, false);
      //if (i == (end_frame-2)) {
      ////if (i == 2*20) {
      //  cv::Mat eq_img;
      //  cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
      //  cv::equalizeHist(eq_img, eq_img);
      //  cv::imwrite("heat_map_with_stats_orig.png", error_img);
      //  cv::imwrite("heat_map_with_stats.png", eq_img);
      //  track::EvalHelper::DrawErrorDistribution(h_bins, v_bins, left_reproj_errors,
      //                                           error_img, false, true);
      //  cv::cvtColor(error_img, eq_img, cv::COLOR_RGB2GRAY);
      //  cv::equalizeHist(eq_img, eq_img);
      //  cv::imwrite("heat_map_orig.png", error_img);
      //  cv::imwrite("heat_map.png", eq_img);
      //}

      //if (have_gt) {
      //  std::cout << "groundtruth:\n" << gt_world_motion[motion_idx] << std::endl;
      //  //double trans_error;
      //  //core::MathHelper::GetMotionError(gt_motion[i/2], Rt, trans_error);
      //  //std::cout << "Translation error = " << trans_error << " m\n";
      //  ////if (trans_error > 0.8) throw 1;
      //  //if (trans_error > max_trans_error)
      //  //  max_trans_error = trans_error;
      //}
      if (debug)
        while(cv::waitKey(0) != 27);

      //track::EvalHelper::DrawTracksAndErrors(*stereo_tracker, cam_params, Rt, img_left_prev,
      //                                       img_right_prev);
      // TODO:
      //Eigen::Vector3d t, r;
      //core::MathHelper::GetEulerAngles(Rt, r);
      //core::MathHelper::GetTranslation(Rt, t);

      // if the motion is to small skip it
      // tracker.undo_state();
      //MathHelper::InverseTransform(Rt, Rt_inv);    // better
      // on success, update current pose
      //pose = pose * Rt_inv;
      pose = pose * Rt.inverse();
      egomotion_Rt_all.push_back(pose);

      if (use_ba) {
        //bundle_adjuster->UpdateTracks(*stereo_tracker, Rt);
        bundle_adjuster->UpdateTracks(*stereo_tracker, Eigen::Matrix4d::Identity());
        //bundle_adjuster_multiframe->update_tracks(*stereo_tracker, Rt);
        // for old BA version
        //bundle_adjuster_multiframe->update_tracks(*stereo_tracker, Rt_inv);
        //if (frame_num >= 2) {
        if (frame_num >= (bundle_adjuster->num_frames() - 1)) {
          //double egomotion_error = track::EvalHelper::
          //    GetStereoReprojError(*stereo_tracker, cam_params, Rt);
          //double gt_error;
          //if (have_gt)
          //  gt_error = track::EvalHelper::
          //      GetStereoReprojError(*stereo_tracker, cam_params, gt_world_motion[motion_idx]);
          if(!bundle_adjuster->Optimize())
            return;
          //cv::Mat Rt_sba = bundle_adjuster->
          //    camera_motion(bundle_adjuster->camera_motion_num()-1);
          //cv::Mat Rt_sba_inv;
          //core::MathHelper::invTrans(Rt_sba, Rt_sba_inv);
          if (use_ba) {
            std::cout << "Two-frame odometry:\n" << Rt << std::endl;
            std::cout << "BA odometry:\n" << bundle_adjuster->camera_motions().back() << "\n";
          }

          //double ba_error = track::EvalHelper::
          //    GetStereoReprojError(*stereo_tracker, cam_params, Rt_sba_inv);
          //if (have_gt) printf("GT reproj error = %e\n", gt_error);
          //printf("egomotion reproj error = %e\n", egomotion_error);
          //printf("BA reproj error = %e\n", ba_error);
        }
        //if (frame_num >= (mf_ba_frames-1)) {
        //  bundle_adjuster_multiframe->optimize();
        //  cv::Mat Rt_sba = bundle_adjuster_multiframe->camera_motion(
        //                bundle_adjuster_multiframe->camera_motion_num() - 1);
        //  cv::Mat Rt_sba_inv;
        //  MathHelper::invTrans(Rt_sba, Rt_sba_inv);
        //  cout << "multi-frame BA odometry:\n" << Rt_sba_inv << endl;
        //}
      }

      //double num_matches = egomotion->getNumberOfMatches();
      //double num_inliers = egomotion->getNumberOfInliers();
      //cout << "[egomotion] Matches: " << num_matches << ", Inliers: " << num_inliers << " -> "
      //     << 100.0*num_inliers/num_matches << " %" << endl;
      std::cout << "-----------------------------------------------------------------------\n\n";
      //cv::Mat location_egomotion(pose_egomotion, Range(0,4), Range(3,4)); // extract 4-th column
    } else {
      std::cout << "Egomotion failed!\n";
      return;
      //throw "Error\n";
    }
  }
  auto end = std::chrono::system_clock::now();
  //auto elapsed = end - start;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
  std::ofstream outfile("times.txt", std::ios_base::app);
  outfile << track_name << ": " << elapsed.count() << " sec\n";

  //if (have_gt)
  //  std::cout << "Max trans error = " << max_trans_error << " m\n";
  std::cout << "Writing track " << track_name << " to files\n";
  std::ofstream egomotion_ofile(output_folder + '/' + track_name + ".txt");

  std::ofstream ba_ofile;
  std::ofstream ba_ofile2;
  //cv::Mat sba_pose;
  if (use_ba) {
    std::string ba_folder = output_folder.substr(0, output_folder.size()-1) + "_ba/";
    //std::string ba_folder2 = output_folder.substr(0, output_folder.size()-1) + "_ba_multiframe/";
    ba_ofile.open(ba_folder + track_name + ".txt");
    //ba_ofile2.open(ba_folder2 + track_name + ".txt");
    //sba_pose = cv::Mat::eye(4, 4, CV_64F);
  }
  pose = Eigen::Matrix4d::Identity();
  std::vector<Eigen::Matrix4d> bundle_adjusted_motions;
  if (use_ba) {
    bundle_adjusted_motions = bundle_adjuster->camera_motions();
    if (egomotion_Rt_all.size() != bundle_adjusted_motions.size()+1) {
      std::cout << "[egomotion.cc] " << egomotion_Rt_all.size() << "!="
                << bundle_adjusted_motions.size()+1 << "\n";
      return;
    }
  }
  for (size_t i = 0; i < egomotion_Rt_all.size(); i++) {
    FormatHelper::WriteMotionToFile(egomotion_Rt_all[i], egomotion_ofile);
    if (use_ba) {
      FormatHelper::WriteMotionToFile(pose, ba_ofile);
      if (i < bundle_adjusted_motions.size())
        pose = pose * bundle_adjusted_motions[i].inverse();
      //FormatHelper::WriteMatRt(bundle_adjuster_multiframe->camera_motion_acc(i), ba_ofile2);
    }
  }

  if (use_ba)
    delete bundle_adjuster;
  delete detector;
  delete mono_tracker;
  delete stereo_tracker;
  delete egomotion;

  //// run evaluation
  //std::string cmd = "/home/kivan/Projects/cv-stereo/stereo_odometry/evaluation/" +
  //                  "evaluate_odometry_dense 7 7 " +
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
  std::string deformation_field_path;
  std::string output_folder;

  if (argc != 5 && argc != 7 && argc != 9){
    std::cout << "Usage:\n" << argv[0]
        << " -c dataset_config -e experiment_config [-o output_folder] [-d deformation_field]\n";
    return 0;
  }

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help, h", "produce help message")
      ("config,c", po::value<std::string>(&config_file), "config filename")
      ("experiment_config,e", po::value<std::string>(&experiment_config), "experiment filename")
      ("output_folder,o", po::value<std::string>(&output_folder), "output folder path")
      ("deformation_field,d", po::value<std::string>(&deformation_field_path),
        "file with deformation field");
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
      ("camera_params,p", po::value<std::string>(&cam_params_file)->
        default_value("camera_params.txt"), "camera params file")
      ("source_folder,s", po::value<std::string>(&source_folder), "folder with source")
      ("imglist,l", po::value<std::string>(&imagelistfn), "file with image list")
      ("groundtruth,g", po::value<std::string>(&gt_filepath), "file with motion GT")
      ("deformation_field,d", po::value<std::string>(&deformation_field_path),
       "file with deformation field");

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
  RunEgomotionExperiment(source_folder, imagelistfn, config_file, experiment_config,
                         cam_params_file, gt_filepath, deformation_field_path, output_folder);

  return 0;
}
