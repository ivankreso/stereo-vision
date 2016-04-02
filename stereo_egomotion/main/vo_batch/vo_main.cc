#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <unistd.h>
#include <sys/wait.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "opencv2/core/core.hpp"
#include "opencv2/core/operations.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

#include "../../../core/image.h"
#include "../../../tracker/stereo/experiment_factory.h"
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
#include "../../../tracker/refiner/feature_refiner_klt.h"
#include "../../../tracker/detector/feature_detector_base.h"
//#include "../../../tracker/descriptor/freak.h"
#include "../../../tracker/stereo/tracker_helper.h"
#include "../../../optimization/sba/sba_base.h"
//#include "../../../optimization/sba/sba_ros.h"
#include "../../../optimization/sba/feature_helper_sba.h"
#include "../../extern/libviso2/src/viso_stereo.h"
#include "../../extern/libviso2/src/matrix.h"
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
  libviso::VisualOdometryStereo* viso = nullptr;
  bool use_ba; // use bundle adjustment
  int ba_frames;
  track::ExperimentFactory::create_experiment(experiment_config, cam_params, output_folder, &detector,
                                              &mono_tracker, &stereo_tracker, &viso, use_ba, ba_frames);

  std::vector<std::string> imagelist;
  bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
    throw "can not open " + imagelistfn + " or the string list is empty\n";

  // TODO
  //param.match.refinement = 2;

  // optional
  //param.inlier_threshold = 0.01;       // 1.5
  //param.ransac_iters = RANSAC_ITERS;           // def: 100
  //param.bucket.bucket_height = 50;    // 50
  //param.bucket.bucket_width = 50;     // 50
  //param.bucket.max_features = 3;      // 2

  int start_frame = 0 * 2;
  //int start_frame = 1110 * 2;
  int end_frame = imagelist.size();
  //int end_frame = 10 * 2;

  bool smooth_images = false;
  cv::Mat img_left_prev, img_right_prev;
  cv::Mat img_left = cv::imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  if(smooth_images) {
    cv::GaussianBlur(img_left, img_left, cv::Size(3,3), 0.7);
    cv::GaussianBlur(img_right, img_right, cv::Size(3,3), 0.7);
  }
  
  optim::FeatureHelperSBA* sba;
  if(use_ba)
    sba = new optim::FeatureHelperSBA(img_left, img_right, ba_frames, stereo_tracker->countFeatures());

  libviso::Matrix pose = libviso::Matrix::eye(4);
  libviso::Matrix viso_cvtrack = libviso::Matrix::eye(4);
  //Matrix point_rt = Matrix::eye(4);
  cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_64F);
  Mat Rt_inv;
  Mat mat_I = Mat::eye(4, 4, CV_64F);
  mat_I.copyTo(Rt_inv);
  Vec<double,7> trans_vec;
  extr_params.push_back(mat_I.clone());
  Mat Rt(4, 4, CV_64F);
  Mat Rt_gt(4, 4, CV_64F);
  Mat Rt_gt_prev = Mat::eye(4, 4, CV_64F);
  Mat Rt_gt_curr = Mat::eye(4, 4, CV_64F);
  MathHelper::invTrans(Rt_inv, Rt);
  Rt_params.push_back(Rt.clone());

  Mat prev_location_viso = Mat::zeros(4, 1, CV_64F);

  // init the tracker
  stereo_tracker->init(img_left, img_right);

  Mat disp_camera_left = Mat::zeros(img_left.rows, img_left.cols, CV_8UC3);
  Mat disp_camera_right = Mat::zeros(img_left.rows, img_left.cols, CV_8UC3);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<cv::Mat> libviso_Rt_all;
  cv::Mat pose_libviso = Mat::eye(4, 4, CV_64F);
  libviso_Rt_all.push_back(pose_libviso.clone());
  
  bool using_kitti = true;
  std::string track_name;
  //std::ifstream gt_file(gt_filepath);
  if(experiment_config.find("tsukuba") != std::string::npos) {
    printf("---------- Using Tsukuba dataset! ------------\n");
    track_name = "00";
    using_kitti = false;
    //gt_file.open("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/tsukuba_gt_crop.txt");
  }
  else {
    track_name = imagelistfn.substr(imagelistfn.size()-10,2);
    //gt_file.open("/home/kivan/Projects/datasets/KITTI/poses/07.txt");
  }

  //int gt_better_cnt = 0;
  int frame_num = start_frame / 2 + 1;
  for(unsigned i = start_frame + 2; i < end_frame; i+=(2)) {
    frame_num = i / 2 + 1;
    std::cout << "Frame: " << frame_num << " / " << imagelist.size()/2 << "\n";
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

    // convert from my tracker to Libviso data structure
    std::vector<int> active_tracks;
    std::vector<libviso::Matcher::p_match> libviso_tracks;
    FeatureHelper::TrackerBaseToLibviso(stereo_tracker, libviso_tracks, active_tracks);

    // run odometry optimization
    if(viso->process(libviso_tracks)) {
      // is it better to kill or not to kill outliers in tracker?
      std::vector<int32_t>& inliers = viso->getInliers();
      std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      //FeatureHelper::FilterOutlierTracks(*stereo_tracker, active_tracks, inliers, 30.0, 0.1);
      FeatureHelper::FilterRansacOutliers(*stereo_tracker, active_tracks, inliers);
      std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";

      //libviso::Matrix Rt_inv_libviso = libviso::Matrix::inv(viso.getMotion());
      //MathHelper::matrixToMat(Rt_inv_libviso, Rt_inv);
      libviso::Matrix Rt_libviso = viso->getMotion();
      MathHelper::matrixToMat(Rt_libviso, Rt);

      // if the motion is to small skip it
      // tracker.undo_state();
      MathHelper::invTrans(Rt, Rt_inv);    // better
      extr_params.push_back(Rt_inv.clone());
      
      // on success, update current pose
      pose_mat = pose_mat * Rt_inv;
      libviso_Rt_all.push_back(pose_mat.clone());

      if(use_ba) {
        // ----- SBA -----
        sba->updateTracks(img_left, img_right, *stereo_tracker, Rt_inv, cam_params);
        //sba->updateTracks(img_left, img_right, *stereo_tracker, mat_I, cam_params);
        if(frame_num >= ba_frames) {
          sba->runSBA();
          cout << "-------------------------------\n";
          //cout << "reprojection error (SBA): " << reproj_error_sba << "\n";
        }
      }

      double num_matches = viso->getNumberOfMatches();
      double num_inliers = viso->getNumberOfInliers();
      cout << "[Libviso] Matches: " << num_matches << ", Inliers: " << num_inliers << " -> "
           << 100.0*num_inliers/num_matches << " %" << endl;
      cout << "-----------------------------------------------------------------------\n\n";
      Mat location_viso(pose_libviso, Range(0,4), Range(3,4)); // extract 4-th column

      location_viso.copyTo(prev_location_viso);
    } else {
      cout << "libviso ... failed!" << endl;
      throw "Error\n";
      waitKey(0);
      //extr_params.push_back(Rt_inv.clone());
      //exit(1);
    }
  }

  //waitKey(0);
  //destroyAllWindows();

  //std::cout << "Groundtruth data cost is better then odometry in " << gt_better_cnt << " / "
  //          << imagelist.size() << " frames.\n";

  std::cout << "Writing track " << track_name << " to files\n";
  std::ofstream libviso_ofile(output_folder + '/' + track_name + ".txt");

  std::ofstream sba_ofile;
  cv::Mat sba_pose;
  if(use_ba) {
    std::string ba_folder = output_folder.substr(0, output_folder.size()-1) + "_ba/";
    sba_ofile.open(ba_folder + track_name + ".txt");
    sba_pose = cv::Mat::eye(4, 4, CV_64F);
  }
  for(size_t i = 0; i < libviso_Rt_all.size(); i++) {
    FormatHelper::writeMatRt(libviso_Rt_all[i], libviso_ofile);
    if(use_ba) {
      sba_pose = sba_pose * sba->getCameraRt(i);
      FormatHelper::writeMatRt(sba_pose, sba_ofile);
    }
  }

  if(use_ba)
    delete sba;
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

  run_visual_odometry(source_folder, imagelistfn, experiment_config, cam_params_file, gt_filepath);

  return 0;
}
