#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <unistd.h>
#include <sys/wait.h>

#include <boost/program_options.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
namespace po = boost::program_options;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../tracker/detector/feature_detector_harris_cv.h"
#include "../../../tracker/mono/tracker_bfm.h"
#include "../../../tracker/stereo/stereo_tracker.h"
#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"
#include "../../../optimization/calibration_bias/deformation_field_solver.h"

void ImportTracks(const std::string& image_list_path,
                  const std::string& tracks_path,
                  const std::string& groudtruth_path,
                  optim::DeformationFieldSolver& solver)
{
  std::vector<std::string> imagelist;
  bool ok = core::FormatHelper::readStringList(image_list_path, imagelist);
  if (!ok || imagelist.empty())
    throw "can not open " + image_list_path + " or the string list is empty\n";

  int start_frame = 0 * 2;
  int end_frame = imagelist.size();

  std::string track_name;
  bool have_gt = false;
  if (!groudtruth_path.empty())
    have_gt = true;
  else throw 1;

  std::vector<cv::Mat> gt_world_motion, gt_camera_motion;
  int num_of_motions = ((imagelist.size()/2) - 1);
  core::FormatHelper::Read2FrameMotionFromAccCameraMotion(groudtruth_path, num_of_motions,
                                                          gt_world_motion, gt_camera_motion);

  track::FeatureDetectorHarrisCV feature_detector(0, 0, 0, 0, 0);
  track::TrackerBFM mono_tracker(feature_detector, 0, 0, 0, 0);
  track::StereoTracker stereo_tracker(mono_tracker, 0, 0, 0, 0, false, "");
  int frame_num = 0;
  std::cout << "Loading tracks: " << tracks_path << "\n";
  for (unsigned i = start_frame + 2; i < end_frame; i += 2) {
    int motion_idx = i / 2 - 1;
    frame_num++;
    //std::cout << "motion_idx: " << motion_idx << " / " << (gt_world_motion.size() - 1) << "\n";
    //std::cout << "Frame num: " << frame_num << "\n";
    //std::cout << "Image num: " << image_num << " / " << end_frame/2-1 << "\n";
    std::string tracker_file =  tracks_path + "/tracks_" + std::to_string(motion_idx);
    //std::cout << tracker_file << "\n";

    // just for KITTI
    //std::cout << "GT trans diff = " << trans_diff << "\n";
    if (tracks_path.find("tsukuba") == std::string::npos) {
      if (frame_num > 1) {
        // diff check to test for estimation chrashes on KITTI test
        double trans_diff;
        core::MathHelper::GetMotionError(gt_world_motion[motion_idx-1], gt_world_motion[motion_idx], trans_diff);
        if (trans_diff > 0.35) {
          std::cout << gt_world_motion[motion_idx-1] << "\n\n" << gt_world_motion[motion_idx] << "\n";
          std::cout << "[Error]: motion_idx: " << motion_idx << " - Big GT trans diff = " << trans_diff << "\n";
          continue;
          //throw 1;
        }
      }
    }

    //stereo_tracker->track(img_left, img_right);
    std::ifstream input_file(tracker_file);
    //boost::archive::text_iarchive iarchive(input_file);
    boost::archive::binary_iarchive iarchive(input_file);
    // write class instance to archive
    iarchive >> stereo_tracker;
    //std::cout << stereo_tracker.countActiveTracks() << " active tracks\n";
    solver.UpdateTracks(stereo_tracker, gt_world_motion[motion_idx]);
    //solver.UpdateReverseTracks(stereo_tracker, gt_camera_motion[motion_idx]);
  }
}


void ImportTracksToSolver(const std::string config_file, const std::string tracks_path,
                          optim::DeformationFieldSolver& solver)
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
    throw 1;
  }

  std::cout << "Using track config = " << config_file << '\n';
  ImportTracks(imagelistfn, tracks_path, gt_filepath, solver);
}


int main(int argc, char* argv[]) {
  // Initialize Google's logging library.
  // http://google-glog.googlecode.com/svn/trunk/doc/glog.html
  google::InitGoogleLogging(argv[0]);

  if (argc != 5)
    std::cout << "Usage:\n" << argv[0] << " -skip seq_num -scale robust_loss_scale\n";
  int skip_seq = std::stoi(argv[2]);
  std::string rls = argv[4];
  double robust_loss_scale = std::stod(rls);

  // 1226 / 370 = 3.3135
  //int bin_rows = 3;
  //int bin_cols = 9;
  //int bin_rows = 5;
  //int bin_cols = 15;
  //int bin_rows = 7;
  //int bin_cols = 23;
  //int bin_rows = 11;
  //int bin_cols = 37;
  //int bin_rows = 15;
  //int bin_cols = 49;
  int bin_rows = 21;
  int bin_cols = 69;

  //std::string cam_params_path = "/home/kivan/Projects/cv-stereo/config_files/camera_params_kitti_00.txt";
  //std::string img_path = "/home/kivan/Projects/datasets/KITTI/sequences_gray/00/image_0/000000.png";

  std::string cam_params_path = "/home/kivan/source/cv-stereo/config_files/camera_params_kitti_04.txt";
  std::string img_path = "/home/kivan/datasets/KITTI/sequences_gray/04/image_0/000000.png";

  double cam_params[5];
  core::FormatHelper::readCameraParams(cam_params_path, cam_params);
  cv::Mat img_left = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
  if (img_left.empty()) {
    std::cout << "Error: no images!\n";
    throw 1;
  }
  int img_rows = img_left.rows;
  int img_cols = img_left.cols;

  //std::string tracks_path_04 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_04/";
  //std::string tracks_path_05 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_05/";
  //std::string tracks_path_06 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_06/";
  //std::string tracks_path_07 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_07/";
  //std::string tracks_path_08 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_08/";
  //std::string tracks_path_09 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_09/";
  //std::string tracks_path_10 = "/mnt/ssd/kivan/datasets/tracker_data/ncc_with_ransac/kitti_10/";

  std::string tracks_path_00 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_00/";
  std::string tracks_path_01 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_01/";
  std::string tracks_path_02 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_02/";

  std::string tracks_path_04 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_04/";
  std::string tracks_path_05 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_05/";
  std::string tracks_path_06 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_06/";
  std::string tracks_path_07 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_07/";
  std::string tracks_path_08 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_08/";
  std::string tracks_path_09 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_09/";
  std::string tracks_path_10 = "/mnt/ssd/kivan/datasets/tracker_data/ncc/kitti_10/";

  // 00 - 02
  //optim::DeformationFieldSolver solver(cam_params, bin_rows, bin_cols, img_rows, img_cols, robust_loss_scale,
  //                                     "00_df.yml");
  //ImportTracksToSolver("/home/kivan/Projects/cv-stereo/config_files/config_kitti_00.txt", tracks_path_00, solver);
  //ImportTracksToSolver("/home/kivan/Projects/cv-stereo/config_files/config_kitti_01.txt", tracks_path_01, solver);
  //ImportTracksToSolver("/home/kivan/Projects/cv-stereo/config_files/config_kitti_02.txt", tracks_path_02, solver);

  //std::string save_file = "without_07_df_" + rls + ".yml";
  //std::string save_file = "without_" + std::to_string(skip_seq) + "_df_" + rls + ".yml";
  std::stringstream seq_num;
  seq_num << std::setfill('0') << std::setw(2) << skip_seq;
  std::string save_file = "without_" + seq_num.str() + "_deformation_field_matrix.yml";
  std::cout << save_file << "\n";
  // 04 - 10
  optim::DeformationFieldSolver solver(cam_params, bin_rows, bin_cols, img_rows, img_cols,
                                       robust_loss_scale, save_file);
                                       //"04_df.yml");
                                       //"without_04_df.yml");
                                       //"without_05_df.yml");
                                       //"without_06_df.yml");
                                       //"without_07_df.yml");
                                       //"without_08_df.yml");
                                       //"without_09_df.yml");

  if (skip_seq != 4)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_04.txt", tracks_path_04, solver);
  if (skip_seq != 5)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_05.txt", tracks_path_05, solver);
  if (skip_seq != 6)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_06.txt", tracks_path_06, solver);
  if (skip_seq != 7)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_07.txt", tracks_path_07, solver);
  if (skip_seq != 8)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_08.txt", tracks_path_08, solver);
  if (skip_seq != 9)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_09.txt", tracks_path_09, solver);
  if (skip_seq != 10)
    ImportTracksToSolver("/home/kivan/source/cv-stereo/config_files/config_kitti_10.txt", tracks_path_10, solver);

  solver.Solve();

  return 0;
}
