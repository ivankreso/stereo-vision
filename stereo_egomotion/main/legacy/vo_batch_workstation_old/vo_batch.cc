#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
using namespace std;

#include <unistd.h>
#include <sys/wait.h>

#include <boost/program_options.hpp>
using namespace boost;
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
                         const std::string& cparams_file)
{
  vector<KeyPoint> features_full;
  vector<KeyPoint> features_left;
  vector<KeyPoint> features_right;

  deque<vector<uchar>> track_status;
  deque<vector<KeyPoint>> features_libviso_left;
  deque<vector<KeyPoint>> features_libviso_right;

  deque<Mat> extr_params; // pose mat
  deque<Mat> Rt_params;

  double cam_params[5];
  FormatHelper::readCameraParams(cparams_file, cam_params);

  std::string output_folder;
  FeatureDetectorBase* detector = nullptr;
  TrackerBase* mono_tracker = nullptr;
  StereoTrackerBase* stereo_tracker = nullptr;
  libviso::VisualOdometryStereo* viso = nullptr;
  bool use_ba; // use bundle adjustment
  int ba_frames;
  track::ExperimentFactory::create_experiment(experiment_config, cam_params, output_folder, &detector, &mono_tracker,
                                              &stereo_tracker, &viso, use_ba, ba_frames);

  std::vector<std::string> imagelist;
  bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
    throw "can not open " + imagelistfn + " or the string list is empty\n";

  int start_frame = 0 * 2;
  int end_frame = imagelist.size();
  //int end_frame = 10 * 2;

  cv::Mat img_left_prev, img_right_prev;
  cv::Mat img_left = cv::imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  
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
  
  std::string track_name;
  std::ifstream gt_file;
  if(experiment_config.find("tsukuba") != std::string::npos) {
    track_name = "00";
    //gt_file.open("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/tsukuba_gt_crop.txt");
  }
  else {
    track_name = imagelistfn.substr(imagelistfn.size()-10,2);
    //gt_file.open("/home/kivan/Projects/datasets/KITTI/poses/" + track_name + ".txt");
  }

  // skip first (identety) matrix
  //FormatHelper::ReadNextRtMatrix(gt_file, Rt_gt_curr);

  //int gt_better_cnt = 0;
  int frame_num;;
  for(unsigned i = start_frame + 2; i < end_frame; i+=(2)) {
    cout << source_folder + imagelist[i] << endl;
    cv::swap(img_left_prev, img_left);
    cv::swap(img_right_prev, img_right);
    img_left = cv::imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    frame_num = i / 2 + 1;
    std::cout << "Frame: " << frame_num << " / " << imagelist.size()/2 << "\n";

    stereo_tracker->track(img_left, img_right);
    std::vector<int> active_tracks;
    std::vector<libviso::Matcher::p_match> libviso_tracks;
    FeatureHelper::TrackerBaseToLibviso(stereo_tracker, libviso_tracks, active_tracks);

    if(viso->process(libviso_tracks)) {
      // is it better to kill or not to kill outliers in tracker?
      std::vector<int32_t>& inliers = viso->getInliers();
      std::cout << "Tracks before RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      FeatureHelper::FilterOutlierTracks(*stereo_tracker, active_tracks, inliers);
      std::cout << "Tracks after RANSAC: " << stereo_tracker->countActiveTracks() << "\n";
      // TODO - dont need this
      // enable this only if the tracker doesn't do this already
      //int bad_cnt = FeatureHelper::filterBadTracks(*stereo_tracker);
      //std::cout << "[VisualOdometry]: filtered bad tracks cnt: " << bad_cnt << "\n";

      //libviso::Matrix Rt_inv_libviso = libviso::Matrix::inv(viso.getMotion());
      //MathHelper::matrixToMat(Rt_inv_libviso, Rt_inv);
      libviso::Matrix Rt_libviso = viso->getMotion();
      MathHelper::matrixToMat(Rt_libviso, Rt);
      // TODO:
      //Eigen::Vector3d t, r;
      //core::MathHelper::GetEulerAngles(Rt, r);
      //core::MathHelper::GetTranslation(Rt, t);

      // if the motion is to small skip it
      // tracker.undo_state();
      MathHelper::invTrans(Rt, Rt_inv);    // better
      extr_params.push_back(Rt_inv.clone());
      // option 2: filter from initial odometry data
      // need to pass Rt - from prev to curr coord
      //cout << "Tracks before filtering: " << tracker.countActiveTracks() << "\n";
      ////TrackerHelper::printTrackerStats(tracker);
      //vector<int> outliers;
      //// use only if libviso is doing the uniform feature selection
      //FeatureHelper::filterOutlierTracks(tracker, Rt, cam_params, outliers, OUTLIER_TRACK_EPS);
      //cout << "Tracks after filtering: " << tracker.countActiveTracks() << "\n";
      //TrackerHelper::printTrackerStats(tracker);

      // on success, update current pose
      // TODO try without inverse
      //pose = pose * libviso::Matrix::inv(viso.getMotion());
      pose_mat = pose_mat * Rt_inv;
      //cout << pose << endl << endl;
      //MathHelper::matrixToMat(pose, pose_libviso);
      //libviso_Rt_all.push_back(pose_libviso.clone());
      libviso_Rt_all.push_back(pose_mat.clone());
      //viso_ptsfile << ~pose.extractCols(std::vector<int>(1,3)) << endl;

      // output some statistics
      //std::vector<libviso::Matcher::p_match>& libviso_features = viso->getFeatures();
      //vector<core::Point> points_lp, points_rp, points_lc, points_rc;
      //FeatureHelper::LibvisoInliersToPoints(libviso_features, inliers, points_lp, points_rp,
      //                                      points_lc, points_rc);
      //cout << points_lp.size() << ", " << points_rp.size() << ", " << points_lc.size() << ", "
      //     << points_rc.size() << endl;
      // TODO
      //Mat C = HelperLibviso::getCameraMatrix(param);
      //cv::Mat C;
      //cout << Rt_inv << "\n";

      // reading the GT trans
      //FormatHelper::ReadNextRtMatrix(gt_file, Rt_gt_curr);
      //Mat Rt_gt_inv = Rt_gt_prev * Rt_gt_curr;
      //MathHelper::invTrans(Rt_gt_inv, Rt_gt);
      //MathHelper::invTrans(Rt_gt_curr, Rt_gt_prev);
      //cout << Rt_gt_inv << endl;
      //cout << "Groundtruth:\n" << Rt_gt << endl;
      //cout << "Odometry:\n" << Rt << endl;
      // get camera Rt in for last movement

      if(use_ba) {
        // ----- SBA -----
        // need to pass Rt_inv - from curr to prev coord
        // TODO
        sba->updateTracks(img_left, img_right, *stereo_tracker, Rt_inv, cam_params);
        //sba->updateTracks(img_left, img_right, *stereo_tracker, mat_I, cam_params);
        if(frame_num >= ba_frames) {
          double libviso_error = EvalHelper::GetStereoReprojError(*stereo_tracker, cam_params, Rt);
          double gt_error = EvalHelper::GetStereoReprojError(*stereo_tracker, cam_params, Rt_gt);
          sba->runSBA();
          cv::Mat Rt_sba = sba->getCameraRt(sba->getNumberOfFrames() - 1);
          //if(frame_num > 10) {
          //  cv::Mat Rt_test = sba->getCameraRt(10);
          //  std::cout << Rt_test << "\n\n------";
          //}
          //cv::Mat Rt_sba_inv;
          //MathHelper::invTrans(Rt_sba, Rt_sba_inv);
          //double ba_error = EvalHelper::GetStereoReprojError(*stereo_tracker, cam_params, Rt_sba_inv);
          //printf("GT reproj error = %e\n", gt_error);
          //printf("Libviso reproj error = %e\n", libviso_error);
          //printf("BA reproj error = %e\n", ba_error);
          //cout << "-------------------------------\n";
          //cout << "reprojection error (SBA): " << reproj_error_sba << "\n";
        }
      }

      //double reproj_error = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc,
      //                                                       C, Rt, param.base);
      //cout << "reprojection error (libviso): " << reproj_error << "\n";

      //reprojerr_file << reproj_error << "\n";

      //double reproj_error_gt = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C,
      //    Rt_gt, param.base);
      ///cout << "reprojection error (groundtruth): " << reproj_error_gt << "\n";
      //reprojerr_file_gt << reproj_error_gt << "\n";
      //if(reproj_error > reproj_error_gt) {
      //  cout << "Found better for GT!!!\n";
      //  gt_better_cnt++;
      //  //waitKey(0);
      //}

      double num_matches = viso->getNumberOfMatches();
      double num_inliers = viso->getNumberOfInliers();
      cout << "[Libviso] Matches: " << num_matches << ", Inliers: " << num_inliers << " -> "
           << 100.0*num_inliers/num_matches << " %" << endl;
      cout << "-----------------------------------------------------------------------\n\n";
      Mat location_viso(pose_libviso, Range(0,4), Range(3,4)); // extract 4-th column
      //cout << location_viso << endl;
      //Matrix location = ~point_rt.extractCols(std::vector<int>(1,3));
      //for(int k = 0; k < 3; k++) location.val[0][k] = - location.val[0][k];
      //cout << point_rt * ~location << endl;
      //cout << location << endl << endl;

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
  
  if(argv != 5) {
    std::cout << "Error, wrong params.\n";
    std::cout << "Usage:\n" << argv[0] << " -c config.txt -e experiment.txt\n";
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
      ("camera_params,p", po::value<string>(&cam_params_file)->default_value("camera_params.txt"), "camera params file")
      ("source_folder,s", po::value<string>(&source_folder), "folder with source")
      ("imglist,l", po::value<string>(&imagelistfn), "file with image list");
    
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

  run_visual_odometry(source_folder, imagelistfn, experiment_config, cam_params_file);

  return 0;
}
