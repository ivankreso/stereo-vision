#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

//#include <unistd.h>
//#include <sys/wait.h>

#include <boost/program_options.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../core/helper.h"
#include "../../../core/math_helper.h"
#include "../../../stereo_egomotion/base/egomotion_base.h"
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

void RunEgomotion(const std::string& dataset_name, const std::string& left_folder,
                  const std::string& right_folder, size_t start_num, size_t end_num,
                  size_t num_width, const std::string& camera_filepath,
                  const std::string& experiment_config, const std::string& deformation_field_path) {
  //bool have_gt = false;
  //std::ifstream gt_file;
  //if (!gt_filepath.empty()) {
  //  have_gt = true;
  //  gt_file.open(gt_filepath);
  //}

  double cam_params[5];
  FormatHelper::readCameraParams(camera_filepath, cam_params);
  Eigen::VectorXd camera_params(5);
  for (int i = 0; i < 5; i++)
    camera_params[i] = cam_params[i];

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

  if (use_ba) {
    bundle_adjuster->SetCameraParams(camera_params);
  }

  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d Rt = Eigen::Matrix4d::Identity();

  // init the tracker
  stereo_tracker->init(img_left, img_right);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<Eigen::Matrix4d> egomotion_Rt_all;
  egomotion_Rt_all.push_back(pose);

  //std::vector<cv::Mat> gt_world_motion, gt_camera_motion;
  //if (have_gt) {
  //  int num_of_motions = ((imagelist.size()/2) - 1);
  //  core::FormatHelper::Read2FrameMotionFromAccCameraMotion(gt_filepath, num_of_motions,
  //                                                          gt_world_motion, gt_camera_motion);
  //}
  size_t frame_num = 0;
  auto start = std::chrono::system_clock::now();
  for (size_t i = start_num+1; i <= end_num; i++) {
    frame_num++;
    std::cout << "Image num: " << i << " / " << end_num << "\n";
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
    double min_disp = 0.1;
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
      std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      std::vector<int> outliers = egomotion->GetTrackerOutliers();
      for (size_t j = 0; j < outliers.size(); j++)
        stereo_tracker->removeTrack(outliers[j]);
      std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";

      //// draw tracks
      if (debug)
        track::EvalHelper::DrawStereoTracks(*stereo_tracker, img_left_prev, img_right_prev,
                                            "ransac_left", "ransac_right");


      //MathHelper::InverseTransform(Rt, Rt_inv);    // better
      // on success, update current pose
      //pose = pose * Rt_inv;
      pose = pose * Rt.inverse();
      egomotion_Rt_all.push_back(pose);

      std::cout << "Two-frame odometry:\n" << Rt << std::endl;
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
      std::cout << "-----------------------------------------------------------------------\n\n";
    } else {
      std::cout << "Egomotion failed!\n";
      throw 1;
    }
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time = " << elapsed.count() << " sec\n";
  std::ofstream outfile("times.txt", std::ios_base::app);
  outfile << dataset_name << ": " << elapsed.count() << " sec\n";

  //if (have_gt)
  //  std::cout << "Max trans error = " << max_trans_error << " m\n";
  std::cout << "Writing track " << dataset_name << " to files\n";
  std::ofstream egomotion_ofile(output_folder + '/' + dataset_name + ".txt");

  std::ofstream ba_ofile;
  std::ofstream ba_ofile2;
  //cv::Mat sba_pose;
  if (use_ba) {
    std::string ba_folder = output_folder.substr(0, output_folder.size()-1) + "_ba/";
    ba_ofile.open(ba_folder + dataset_name + ".txt");
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
}

int main(int argc, char** argv)
{
  std::string config_file;
  std::string experiment_config;
  std::string cam_params_file;
  std::string dataset_name;
  std::string left_folder;
  std::string right_folder;
  std::string gt_filepath;
  std::string deformation_field_path;
  std::string output_folder;
  std::size_t start_num, end_num, num_width;

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
      ("dataset_name", po::value<std::string>(&dataset_name)->required(), "name")
      ("camera_params,p", po::value<std::string>(&cam_params_file)->
        default_value("camera_params.txt"), "camera params file")
      ("left_folder", po::value<std::string>(&left_folder)->required(), "folder")
      ("right_folder", po::value<std::string>(&right_folder)->required(), "folder")
      ("start_num", po::value<std::size_t>(&start_num)->required(), "start number")
      ("end_num", po::value<std::size_t>(&end_num)->required(), "end number")
      ("num_width", po::value<std::size_t>(&num_width)->required(), "padding")
      ("groundtruth,g", po::value<std::string>(&gt_filepath), "file with motion GT");

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
  RunEgomotion(dataset_name, left_folder, right_folder, start_num, end_num, num_width,
               cam_params_file, experiment_config, deformation_field_path);

  return 0;
}
