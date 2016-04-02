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
#include "../../../tracker/stereo/stereo_tracker_base.h"
#include "../../../tracker/stereo/stereo_tracker_sim.h"
#include "../../../tracker/stereo/stereo_tracker_libviso.h"
#include "../../../tracker/stereo/stereo_tracker_bfm.h"
#include "../../../tracker/stereo/stereo_tracker.h"
#include "../../../tracker/stereo/stereo_tracker_refiner.h"
#include "../../../tracker/stereo/tracker_refiner_libviso.h"
#include "../../../tracker/refiner/feature_refiner_base.cc"
#include "../../../tracker/refiner/feature_refiner_klt.h"
#include "../../../tracker/detector/feature_detector_harris_cv.h"
#include "../../../tracker/detector/feature_detector_uniform.h"
#include "../../../tracker/detector/feature_detector_gftt_cv.h"
#include "../../../tracker/detector/feature_detector_base.h"
//#include "../../../tracker/descriptor/freak.h"
#include "../../../tracker/stereo/tracker_helper.h"
#include "../../../optimization/sba/sba_base.h"
#include "../../../optimization/sba/sba_ros.h"
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

// plotting settings
#define SCALE_FACTOR            1.0       // 0.5, zoom with 2.0, tsukuba - 0.3
#define CENTER_X                300
#define CENTER_Y                200
#define WINDOW_WIDTH            900
#define WINDOW_HEIGHT           700
#define MAX_DISP_CHANGE         5

//TODO hardcoded - in future add movement checking through optical flow
// feature tracking params
#define SBA_FRAME_NUM           2  // 3-10

// matcher params
#define SEARCH_WINDOW             180            // KITTI - 180, BB2 - 120, TSUKUBA - 140, iva - 300
#define MIN_NCC                   0.9         // 0.8 - 0.9,     iva - 0.7
#define NCC_PATCH_SIZE            15          // 11 (Nister),  21(Irschara), 15(Refiner)
#define MAX_HAMM                  80          // FREAK - 60
#define MAX_FEATURES              4000        // 1000 - 5000, iva - 20000
//#define OUTLIER_TRACK_EPS         2.5 // 2.0

// detector params
#define HARRIS_BLOCK_SIZE         3     // 3, irchara uses 5 but more matches with 3
#define HARRIS_FILTER_SIZE        3     // 3, or 1
#define HARRIS_K                  0.04        // Nister - 0.06
// TODO:
#define HARRIS_THR                0.0000001         // 0.000001, iva - 0.0000001
//this depends on feature destcriptor used
//#define HARRIS_MARGIN             (NCC_PATCH_SIZE / 2)     // NCC_PATCH_SIZE/2 for NCC, 67 for FREAK
// NCC_PATCH_SIZE+1/2 for NCC - +1 because of gradient image in FeatureRefinerKLT, 67 for FREAK
#define HARRIS_MARGIN             ((NCC_PATCH_SIZE+1) / 2)
#define HORIZ_BINS                10
#define VERTI_BINS                10
#define FEATURES_PER_BIN          20    // 20, try 15 for BB

//#define GFFT_MIN_DISTANCE         0.0       // 0, 0-5
//#define GFFT_QLT_LEVEL            0.00000000001 // 0.0001, 0.00001
//#define GFFT_MAX_CORNERS          10000     // 5000, 2000 -

//string disp_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/disparity_maps/left/");
string depth_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/depth_maps/left/");
string depthfn("/home/kivan/Projects/cv-stereo/config_files/tsukuba_depth_lst.xml");
std::vector<std::string> depth_filelist;

void visualOdometry(string source_folder, vector<string>& imagelist, string cparams_file, string output_folder)
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
  libviso::VisualOdometryStereo::parameters param;
  param.calib.f = cam_params[0];  // focal length in pixels
  param.calib.cu = cam_params[2]; // principal point (u-coordinate) in pixels
  param.calib.cv = cam_params[3]; // principal point (v-coordinate) in pixels
  param.base = cam_params[4];

  // TODO
  //param.match.refinement = 2;

  // optional
  //param.inlier_threshold = 0.01;       // 1.5
  //param.ransac_iters = 200;           // 200
  //param.bucket.bucket_height = 50;    // 50
  //param.bucket.bucket_width = 50;     // 50
  //param.bucket.max_features = 3;      // 2

  //string simdata_folder = "/home/kivan/projects/datasets/stereo_model/points_kitti_cam_nonoise_base_0.50/";
  //string simdata_xml = "/home/kivan/projects/datasets/stereo_model/stereosim_viso00path.xml";
  //Mat mask = imread("/home/kivan/Projects/cv-stereo/config_files/mask_kitti_rect.png",
  //                  CV_LOAD_IMAGE_GRAYSCALE);

  ifstream groundtruth_file("/home/kivan/Projects/datasets/KITTI/poses/07.txt");
  string src_disp("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/disparity_maps/");
  string dispfn("/home/kivan/Projects/cv-stereo/config_files/truskuba_disp_lst.xml");

  //StereoTrackerBase* tracker = new StereoTrackerSim(simdata_folder, simdata_xml);
  FeatureRefinerKLT refiner;

  //FeatureDetectorGFTTCV detector(GFFT_MAX_CORNERS, GFFT_QLT_LEVEL, GFFT_MIN_DISTANCE, HARRIS_BLOCK_SIZE,
  //                               USE_HARRIS, HARRIS_K, mask, HORIZ_BINS, VERTI_BINS, FEATURES_PER_BIN);
  //FeatureDetectorHarrisCV detector_basic(HARRIS_BLOCK_SIZE, HARRIS_FILTER_SIZE, HARRIS_K,
  //                                       HARRIS_THR, HARRIS_MARGIN);
  //FeatureDetectorUniform detector(detector_basic, HORIZ_BINS, VERTI_BINS, FEATURES_PER_BIN);
  FeatureDetectorHarrisCV detector(HARRIS_BLOCK_SIZE, HARRIS_FILTER_SIZE, HARRIS_K, HARRIS_THR, HARRIS_MARGIN);

  //cv::FREAK dextractor;
  //cv::FREAK dextractor(false, false);
  //StereoTracker tracker(detector, dextractor, MAX_FEATURES, MAX_HAMM, SEARCH_WINDOW);

  //StereoTrackerBFM tracker(&detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SEARCH_WINDOW, MAX_DISP_CHANGE);
  StereoTrackerBFM tracker_basic(&detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SEARCH_WINDOW, MAX_DISP_CHANGE);
  StereoTrackerRefiner tracker(&tracker_basic, &refiner);

  // not using this anymore - TrackerRefinerLibviso tracker(&tracker_basic, &refiner);

  bool is_libviso_refiner = false;
  //StereoTrackerLibviso tracker(param);

  //is_libviso_refiner = true;
  //StereoTrackerLibviso tracker_basic(param, MAX_FEATURES);
  //StereoTrackerRefiner tracker(&tracker_basic, &refiner);

  // init visual odometry
  libviso::VisualOdometryStereo viso(param, &tracker);

  //optim::FeatureHelperSBA sba(SBA_FRAME_NUM, MAX_FEATURES);

  int start_frame = 0 * 2;
  //int start_frame = 186 * 2;
  //Mat descriptors_1, descriptors_2;
  Mat img_left_prev = imread(source_folder + imagelist[start_frame], CV_LOAD_IMAGE_GRAYSCALE);
  Mat img_right_prev = imread(source_folder + imagelist[start_frame+1], CV_LOAD_IMAGE_GRAYSCALE);
  //cv::Mat disp_imt = imread(;
  cv::Mat mat_depth;
  cv::FileStorage fs(depth_folder + depth_filelist[start_frame/2], cv::FileStorage::READ);
  cout << depth_folder + depth_filelist[start_frame/2] << endl;
  fs["depth"] >> mat_depth;

  //Image tmp_img;
  //HelperOpencv::MatToImage(img_left_prev, tmp_img);
  //ImageSetExact imgset;
  //imgset.compute(tmp_img);
  // first calc margins size of gradient images
  //const int gradient_margin = (imgset.kernelGrad_.size() / 2) + 1;
  //// calculate feature detector mask using feature size and extra gradient margin size
  //const int fhw = (FeatureData::width() / 2) + gradient_margin;
  //const int fhh = (FeatureData::height() / 2) + gradient_margin;
  //detector.setMask(fhw, fhh, img_left_prev.cols - fhw, img_left_prev.rows - fhh);

  libviso::Matrix pose = libviso::Matrix::eye(4);
  libviso::Matrix viso_cvtrack = libviso::Matrix::eye(4);
  //Matrix point_rt = Matrix::eye(4);
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
  Mat prev_location_viso_cvtrack = Mat::zeros(4, 1, CV_64F);
  Mat prev_location_sba = Mat::zeros(4, 1, CV_64F);

  int32_t dims[] = {img_left_prev.cols, img_left_prev.rows, img_left_prev.cols};
  // init the tracker
  if(!viso.process(img_left_prev.data, img_right_prev.data, dims))
    cout << "[Libviso] init frame - no estimation" << endl;

  Mat disp_libviso = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
  Mat disp_libviso_cvtracker = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
  Mat disp_libviso_birch = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
  Mat disp_sba = Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

  Mat disp_camera_left = Mat::zeros(img_left_prev.rows, img_left_prev.cols, CV_8UC3);
  Mat disp_camera_right = Mat::zeros(img_left_prev.rows, img_left_prev.cols, CV_8UC3);

  //imshow("libviso_orig", disp_libviso);
  //imshow("libviso_cvtracker", disp_libviso_cvtracker);
  //imshow("SBA", disp_sba);
  //moveWindow("libviso_orig", 0, 0);
  //moveWindow("libviso_cvtracker", 682, 0);

  // matrix transforms points in current camera coordinate system to world coord system
  std::vector<cv::Mat> libviso_Rt_all;
  Mat pose_libviso = Mat::eye(4, 4, CV_64F);
  libviso_Rt_all.push_back(pose_libviso.clone());

  CvPlotter plotter(WINDOW_WIDTH, WINDOW_HEIGHT, SCALE_FACTOR , CENTER_X, CENTER_Y);
  // skip first (identety) matrix
  // FormatHelper::readNextMatrixKitti(groundtruth_file, Rt_gt_curr);

  // -- demo start
  //string left_out_folder = "/home/kivan/Projects/demo/vo_demo/left_cam/";
  //string right_out_folder = "/home/kivan/Projects/demo/vo_demo/right_cam/";
  //std::ostringstream sframe_num;
  //sframe_num << std::setw(6) << std::setfill('0') << 0;
  //cout << sframe_num.str() << "\n";
  //imwrite(left_out_folder + "img_left_" + sframe_num.str() + ".jpg", img_left_prev);
  //imwrite(right_out_folder + "img_right_" + sframe_num.str() + ".jpg", img_right_prev);
  // -- demo end

  //ofstream reprojerr_file("reproj_error.txt");
  ofstream deptherr_file("depth_error.txt");
  //ofstream reprojerr_file_gt("reproj_error_gt.txt");

  int gt_better_cnt = 0;
  int frame_num = 0;
  for(unsigned i = start_frame + 2; i < imagelist.size(); i+=(2)) {
    cout << source_folder + imagelist[i] << endl;
    Mat img_left = imread(source_folder + imagelist[i], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_right = imread(source_folder + imagelist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    frame_num = i / 2 + 1;
    cout << frame_num << " frame:" << endl;

    //if(frame_num == 185) std::cout << "\n";

    if(viso.process(img_left.data, img_right.data, dims)) {
      vector<libviso::Matcher::p_match>& libviso_features = viso.getFeatures();
      vector<int32_t>& inliers = viso.getInliers();
      std::vector<core::Point> pts_lp, pts_rp, pts_lc, pts_rc;
      HelperLibviso::LibvisoInliersToPoints(libviso_features, inliers, pts_lp, pts_rp, pts_lc, pts_rc);
      double triang_error = track::EvalHelper::getStereoDepthError(tracker, cam_params, mat_depth);
      //double triang_error = core::EvalHelper::getStereoDepthError(tracker, cam_params, mat_depth,
      //                                        img_left_prev, img_right_prev, img_left, img_right);
      //double triang_error = core::EvalHelper::getStereoRefinerDepthError(tracker_basic, tracker, cam_params, mat_depth,
      //                                        img_left_prev, img_right_prev, img_left, img_right);
      //double triang_error = core::EvalHelper::getStereoDepthError(pts_lc, pts_rc,cam_params, mat_depth,
      //                                                            img_left_prev, img_right_prev, img_left, img_right);
      std::cout << "Stereo triang mean abs error: " << triang_error << "\n";
      deptherr_file << triang_error << "\n";
      // read next depth GT data
      fs.open(depth_folder + depth_filelist[i/2], cv::FileStorage::READ);
      fs["depth"] >> mat_depth;

      //cout << "point_rt - rt * viso.getMotion:\n" << point_rt_mat << endl;
      //libviso::Matrix Rt_inv_libviso = libviso::Matrix::inv(viso.getMotion());
      //MathHelper::matrixToMat(Rt_inv_libviso, Rt_inv);
      libviso::Matrix Rt_libviso = viso.getMotion();
      MathHelper::matrixToMat(Rt_libviso, Rt);
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

      // draw tracks
      //cvtColor(img_left_prev, disp_camera_left, COLOR_GRAY2RGB);
      //cvtColor(img_right_prev, disp_camera_right, COLOR_GRAY2RGB);
      //FeatureHelper::drawStereoTracks(tracker, disp_camera_left, disp_camera_right);
      //FeatureHelper::drawStereoRefinerTracks(tracker_basic, tracker, disp_camera_left, disp_camera_right);
      tracker.debug();
      waitKey(0);
      //waitKey(0);

      // if using libviso tracker with refiner - clean all the refiner tracks now
      if(is_libviso_refiner == true) {
        for(int j = 0; j < tracker.countFeatures(); j++) {
          tracker.removeTrack(j);
        }
      }

      // save imgs for demo
      // -- demo start
      //sframe_num.str("");
      //sframe_num << std::setw(6) << std::setfill('0') << (i/2);
      //imwrite(left_out_folder + "img_left_" + sframe_num.str() + ".jpg", disp_camera_left);
      //imwrite(right_out_folder + "img_right_" + sframe_num.str() + ".jpg", disp_camera_right);
      // -- demo end

      //// draw the outliers
      //cvtColor(img_left_prev, disp_camera_left, COLOR_GRAY2RGB);
      //cvtColor(img_right_prev, disp_camera_right, COLOR_GRAY2RGB);
      //vector<int> outliers = viso.getOutliers();
      //FeatureHelper::drawStereoTracks(tracker, outliers, disp_camera_left, disp_camera_right);
      //waitKey(0);

      //vector<vector<KeyPoint>> matches, matches_all;
      //HelperLibviso::convertInlierMatchesToKeys(libviso_features, inliers, matches);
      ////HelperLibviso::convertAllMatchesToKeys(libviso_features, matches_all);
      //Mat disp_allfeats;
      //vector<uchar> viso_status(matches[0].size(), 1);
      //cvtColor(img_left_prev, disp_camera, COLOR_GRAY2RGB);
      //cvtColor(img_left_prev, disp_allfeats, COLOR_GRAY2RGB);
      //HelperLibviso::drawOpticalFlow(disp_camera, matches[0], matches[2], viso_status, Scalar(0,0,255));
      ////FeatureFilter::drawOpticalFlow(disp_allfeats, matches_all[0], matches_all[2], viso_status, Scalar(0,0,255));
      //imshow("camera_left", disp_camera);
      //imshow("camera_left_all", disp_allfeats);
      //moveWindow("camera left", 400, 200);

      // on success, update current pose
      // TODO try without inverse
      pose = pose * libviso::Matrix::inv(viso.getMotion());
      //cout << pose << endl << endl;
      MathHelper::matrixToMat(pose, pose_libviso);
      libviso_Rt_all.push_back(pose_libviso.clone());
      //viso_ptsfile << ~pose.extractCols(std::vector<int>(1,3)) << endl;

      // output some statistics
      vector<core::Point> points_lp, points_rp, points_lc, points_rc;
      FeatureHelper::LibvisoInliersToPoints(libviso_features, inliers, points_lp, points_rp,
                                            points_lc, points_rc);
      //cout << points_lp.size() << ", " << points_rp.size() << ", " << points_lc.size() << ", "
      //     << points_rc.size() << endl;
      Mat C = HelperLibviso::getCameraMatrix(param);
      //cout << Rt_inv << "\n";

      // reading the GT trans
      //FormatHelper::readNextMatrixKitti(groundtruth_file, Rt_gt_curr);
      //Mat Rt_gt_inv = Rt_gt_prev * Rt_gt_curr;
      //MathHelper::invTrans(Rt_gt_inv, Rt_gt);
      //MathHelper::invTrans(Rt_gt_curr, Rt_gt_prev);

      //cout << Rt_gt_inv << endl;
      //cout << "Groundtruth:\n" << Rt_gt << endl;
      //cout << "Odometry:\n" << Rt << endl;
      // get camera Rt in for last movement

      // ----- SBA -----
      // need to pass Rt_inv - from curr to prev coord
      //sba.updateTracks(tracker, Rt_inv, cam_params);
      //if(frame_num >= SBA_FRAME_NUM) {
      //  sba.runSBA();
      //  cv::Mat Rt_sba = sba.getCameraRt(sba.getNumberOfFrames() - 1);
      //  cv::Mat Rt_sba_inv;
      //  MathHelper::invTrans(Rt_sba, Rt_sba_inv);
      //  cout << "-------------------------------\n";
      //  double reproj_error_sba = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C, 
      //      Rt_sba_inv, param.base);
      //  cout << "reprojection error (SBA): " << reproj_error_sba << "\n";
      //}

      double reproj_error = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C,
                                                             Rt, param.base);
      cout << "reprojection error (libviso): " << reproj_error << "\n";
      //reprojerr_file << reproj_error << "\n";

      //double reproj_error_gt = EvalHelper::getStereoReprojError(points_lp, points_rp, points_lc, points_rc, C,
      //    Rt_gt, param.base);
      //cout << "reprojection error (groundtruth): " << reproj_error_gt << "\n";
      //reprojerr_file_gt << reproj_error_gt << "\n";
      //if(reproj_error > reproj_error_gt) {
      //  cout << "Found better for GT!!!\n";
      //  gt_better_cnt++;
      //  //waitKey(0);
      //}
      cout << "-----------------------------------------------------------------------\n";

      double num_matches = viso.getNumberOfMatches();
      double num_inliers = viso.getNumberOfInliers();
      cout << "[Libviso] Matches: " << num_matches << ", Inliers: " << num_inliers << " -> "
           << 100.0*num_inliers/num_matches << " %" << endl;
      Mat location_viso(pose_libviso, Range(0,4), Range(3,4)); // extract 4-th column
      //cout << location_viso << endl;
      //Matrix location = ~point_rt.extractCols(std::vector<int>(1,3));
      //for(int k = 0; k < 3; k++) location.val[0][k] = - location.val[0][k];
      //cout << point_rt * ~location << endl;
      //cout << location << endl << endl;

      // ploting the path
      plotter.drawLine(prev_location_viso, location_viso, disp_libviso);
      imshow("libviso_orig", disp_libviso);
      //waitKey(10);

      location_viso.copyTo(prev_location_viso);
    } else {
      cout << "libviso ... failed!" << endl;
      waitKey(0);
      //extr_params.push_back(Rt_inv.clone());
      exit(1);
    }

    cv::swap(img_left_prev, img_left);
    cv::swap(img_right_prev, img_right);
  }

  destroyAllWindows();
  waitKey(0);

  std::cout << "Groundtruth data cost is better then odometry in " << gt_better_cnt << " / " 
            << imagelist.size() << " frames.\n";
  // save the data results
  //int sba_frames = sba.getNumberOfFrames();
  //assert(libviso_Rt_all.size() == sba_frames);
  ofstream libviso_ofile("vo.txt");
  //ofstream sba_ofile("sba_odom.txt");
  cv::Mat sba_pose = cv::Mat::eye(4, 4, CV_64F);
  for(int i = 0; i < libviso_Rt_all.size(); i++) {
    FormatHelper::writeMatRt(libviso_Rt_all[i], libviso_ofile);
    //sba_pose = sba_pose * sba.getCameraRt(i);
    //FormatHelper::writeMatRt(sba_pose, sba_ofile);
  }

}


int main(int argc, char** argv)
{
  string config_file;
  string imagelistfn;
  string cam_params_file;
  string source_folder;
  string output_folder;

  try {
    po::options_description generic("Generic options");
    generic.add_options()
      ("help", "produce help message")
      ("config,c", po::value<string>(&config_file)->default_value("config.txt"), "config filename")
      ;
    po::options_description config("Config file options");
    config.add_options()
      ("camera_params,p", po::value<string>(&cam_params_file)->default_value("camera_params.txt"), "camera params file")
      ("source_folder,s", po::value<string>(&source_folder), "folder with source")
      //(",o", po::value<string>(&output_folder), "folder for output")
      ("imglist,l", po::value<string>(&imagelistfn), "file with image list")
      ;

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config);

    po::options_description config_file_options;
    config_file_options.add(config);
    po::variables_map vm;
    //po::store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
    if(vm.count("help")) {
      cout << generic;
      cout << config;
      return 0;
    }

    ifstream ifs(config_file.c_str());
    if (!ifs) {
      cout << "can not open config file: " << config_file << "\n";
      cout << generic;
      cout << config;
      return 0;
    }
    else {
      po::store(parse_config_file(ifs, config_file_options, true), vm);
      notify(vm);
    }
    cout << "Configuring done, using:" << endl;

    if(vm.count("camera_params")) {
      cout << "Camera params: ";
      cout << cam_params_file << endl;
    }
    if(vm.count("source_folder")) {
      cout << "Source folder: ";
      cout << source_folder << endl;
    }
    if(vm.count("output_folder")) {
      cout << "Output folder: ";
      cout << output_folder << endl;
    }
    if(vm.count("imglist")) {
      cout << "Image list file: ";
      cout << imagelistfn << endl;
    }
  }
  catch(std::exception& e) {
    cout << e.what() << "\n";
    return 1;
  }

  if(imagelistfn == "")
  {
    cout << "error: no xml image list given." << endl;
    return -1;
  }
  if(source_folder == "")
  {
    cout << "error: no source folder given." << endl;
    return -1;
  }

  vector<string> imagelist;
  bool ok = FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
    cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }

  ok = FormatHelper::readStringList(depthfn, depth_filelist);
  visualOdometry(source_folder, imagelist, cam_params_file, output_folder);

  return 0;
}
