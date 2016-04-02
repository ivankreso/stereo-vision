#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../../../core/image.h"
#include "../../../core/format_helper.h"
#include "../../base/eval_helper.h"
#include "../../mono/tracker_base.h"
#include "../../mono/tracker_opencv.h"
#include "../../mono/tracker_birch.h"
#include "../../mono/tracker_stm.h"
#include "../../mono/tracker_bfm_cv.h"
#include "../../mono/tracker_bfm.h"
#include "../../base/helper_opencv.h"
#include "../../stereo/stereo_tracker_base.h"
#include "../../stereo/stereo_tracker.h"
#include "../../stereo/stereo_tracker_bfm.h"
#include "../../stereo/tracker_helper.h"
#include "../../detector/feature_detector_harris_cv.h"
#include "../../detector/feature_detector_harris_freak.h"
#include "../../detector/feature_detector_gftt_cv.h"

#define MAX_FEATURES              4096        // Kanade - 4096
#define SEARCH_WINDOW             100         // KITTI - 180-200, BB2 - 120, TSUKUBA - 100
#define NCC_THRESHOLD             0.9        // 0.8 - 0.9
#define NCC_WSZ                   15
#define HAMMING_THRESHOLD         50          // 50
#define MAX_DISPARITY             128
#define STEREO_WSZ                NCC_WSZ     // Kanade - 15 with NCC

// detector params
#define HARRIS_BLOCK_SIZE         3     // 3, irchara uses 5 but more matches with 3
#define HARRIS_FILTER_SIZE        3     // 3, or 1
#define HARRIS_K                  0.06        // default: 0.04, Nister - 0.06
#define HARRIS_THR                0.0000001         // more corners - better
//#define HARRIS_THR                0.000001         // less - worse
#define HARRIS_MARGIN             STEREO_WSZ     // NCC_PATCH_SIZE/2 for NCC, 67 for FREAK

// FREAK params
// TODO: try to setup keypoint sizes to 10 or STEREO_WSZ and turn this on
#define NORMALIZE_SCALE           false        // can't do this for Harris
#define NORMALIZE_ORIENTATION     true        // more tracks when turned off
#define NUM_OCTAVES               1           // 1, or try 0 but is should fail...

#define TSUKUBA
#ifdef TSUKUBA
std::string depth_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/depth_maps/left/");
std::string depthfn("/home/kivan/Projects/cv-stereo/config_files/tsukuba_depth_lst.xml");
std::string cparams_file("/home/kivan/Projects/cv-stereo/config_files/camera_params_tsukuba.txt");
std::ifstream groundtruth_file("/home/kivan/Projects/cv-stereo/scripts/converters/tsukuba_gt.txt");
std::vector<std::string> depth_filelist;
#endif

//void draw_matches(track::StereoTrackerBase& tracker, cv::Mat& img_lp, cv::Mat& img_rp, cv::Mat& img_lc, cv::Mat& img_rc)
//{
//  cv::Point2f pt1, pt2;
//  Scalar color(0,0,0);
//  for(int i = 0; i < tracker.countFeatures(); i++) {
//    FeatureInfo feat_left = tracker.featureLeft(i);
//    FeatureInfo feat_right = tracker.featureRight(i);
//    //cout << feat_left.status_ << endl;
//    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
//      Mat img_lp_orig = img_lp.clone();
//      Mat img_rp_orig = img_rp.clone();
//      Mat img_lc_orig = img_lc.clone();
//      Mat img_rc_orig = img_rc.clone();
//      color = Scalar(255,0,0);
//      pt1.x = feat_left.prev_.x_;
//      pt1.y = feat_left.prev_.y_;
//      pt2.x = feat_left.curr_.x_;
//      pt2.y = feat_left.curr_.y_;
//      //cv::line(img_lc, pt1, pt2, color, 2, 8);
//      cv::circle(img_lp, pt1, 4, color, -1, 8);
//      cv::circle(img_lc, pt2, 4, color, -1, 8);
//      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
//
//      pt1.x = feat_right.prev_.x_;
//      pt1.y = feat_right.prev_.y_;
//      pt2.x = feat_right.curr_.x_;
//      pt2.y = feat_right.curr_.y_;
//      //cv::line(img_lc, pt1, pt2, color, 2, 8);
//      cv::circle(img_rp, pt1, 4, color, -1, 8);
//      cv::circle(img_rc, pt2, 4, color, -1, 8);
//      cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
//      imshow("left_prev_track", img_lp);
//      imshow("left_curr_track", img_lc);
//      imshow("right_prev_track", img_rp);
//      imshow("right_curr_track", img_rc);
//      waitKey(0);
//      img_lp = img_lp_orig;
//      img_lc = img_lc_orig;
//      img_rp = img_rp_orig;
//      img_rc = img_rc_orig;
//    }
//  }
//}

void draw_flow(track::StereoTrackerBase& tracker, cv::Mat& img_lc, cv::Mat& img_rc)
{
  cv::Point pt1, pt2;
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat_left = tracker.featureLeft(i);
    track::FeatureInfo feat_right = tracker.featureRight(i);
    //cout << feat_left.status_ << endl;
    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      pt1.x = feat_left.prev_.x_;
      pt1.y = feat_left.prev_.y_;
      pt2.x = feat_left.curr_.x_;
      pt2.y = feat_left.curr_.y_;
      cv::line(img_lc, pt1, pt2, color_prev, 2, 8);
      cv::circle(img_lc, pt1, 2, color_prev, -1, 8);
      cv::circle(img_lc, pt2, 2, color_curr, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      cv::putText(img_lc, std::to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar::all(0), 1, 8);

      pt1.x = feat_right.prev_.x_;
      pt1.y = feat_right.prev_.y_;
      pt2.x = feat_right.curr_.x_;
      pt2.y = feat_right.curr_.y_;
      cv::line(img_rc, pt1, pt2, color_prev, 2, 8);
      cv::circle(img_rc, pt1, 2, color_prev, -1, 8);
      cv::circle(img_rc, pt2, 2, color_curr, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      cv::putText(img_rc, std::to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar::all(0), 1, 8);
    }
  }
  cv::imshow("left_prev_track", img_lc);
  cv::imshow("right_prev_track", img_rc);
}


void run_tracker(std::vector<std::string>& imglist, std::string& source_folder)
{
  int start_index = 0;

#ifdef TSUKUBA
  double cam_params[5];
  core::FormatHelper::readCameraParams(cparams_file, cam_params);  
  if(depthfn.length() > 0)
    core::FormatHelper::readStringList(depthfn, depth_filelist);
  cv::Mat Rt_prev, Rt_curr;
  core::FormatHelper::ReadNextRtMatrix(groundtruth_file, Rt_curr);
  cv::Mat depth_prev, depth_curr;
  cv::FileStorage fs(depth_folder + depth_filelist[start_index/2], cv::FileStorage::READ);
  std::cout << depth_folder + depth_filelist[start_index/2] << "\n";
  fs["depth"] >> depth_curr;
#endif

  cv::Mat img_left = cv::imread(source_folder + imglist[start_index], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right = cv::imread(source_folder + imglist[start_index+1], CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_left_prev, img_right_prev;
  
  cv::Mat disp_left_prev, disp_right_prev, disp_left_curr, disp_right_curr;
  cv::Mat disp_features_lp, disp_features_rp, disp_features_lc, disp_features_rc;
  //cvtColor(cvimg_left, disp_left_track, COLOR_GRAY2RGB);
  //imshow("left_track", disp_left_track);
  //waitKey(0);

  track::FeatureDetectorHarrisCV detector_base(HARRIS_BLOCK_SIZE, HARRIS_FILTER_SIZE, HARRIS_K,
                                               HARRIS_THR, HARRIS_MARGIN);
  // les matches
  cv::FREAK extractor(true, false, 22.0f, 0);
  // more matches
  //cv::FREAK extractor(false, false, 22.0f, 0);
  track::FeatureDetectorHarrisFREAK detector(detector_base, extractor);


  //track::TrackerBFMcv tracker_bfm(detector, MAX_FEATURES, SEARCH_WINDOW, HAMMING_THRESHOLD);
  track::TrackerBFM* tracker_stm = new track::TrackerBFM(detector_base, MAX_FEATURES, NCC_THRESHOLD, NCC_WSZ, SEARCH_WINDOW);
  track::TrackerSTM tracker_mono(tracker_stm);

  //track::TrackerBFM tracker_mono(detector_base, MAX_FEATURES, NCC_THRESHOLD, NCC_WSZ, SEARCH_WINDOW);
  //track::TrackerBFMcv tracker_mono(detector, MAX_FEATURES, SEARCH_WINDOW, HAMMING_THRESHOLD);

  track::StereoTracker tracker(tracker_mono, MAX_DISPARITY, STEREO_WSZ, NCC_THRESHOLD);
  tracker.init(img_left, img_right);

  for(size_t i = start_index+2; i < imglist.size(); i+=2) {
    img_left_prev = img_left;
    img_right_prev = img_right;
    img_left = cv::imread(source_folder + imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
    img_right = cv::imread(source_folder + imglist[i+1], CV_LOAD_IMAGE_GRAYSCALE);

    // debug features
    //vector<core::Point> points;
    //detector->detect(img_left_prev, points);
    //cvtColor(cvimg_left_prev, disp_features_lp, COLOR_GRAY2RGB);
    //DebugHelper::drawFeatures(points, Scalar(255,0,0), disp_features_lp);
    //detector->detect(img_right_prev, points);
    //cvtColor(cvimg_right_prev, disp_features_rp, COLOR_GRAY2RGB);
    //DebugHelper::drawFeatures(points, Scalar(255,0,0), disp_features_rp);
    //detector->detect(img_left, points);
    //cvtColor(cvimg_left, disp_features_lc, COLOR_GRAY2RGB);
    //DebugHelper::drawFeatures(points, Scalar(255,0,0), disp_features_lc);
    //detector->detect(img_right, points);
    //cvtColor(cvimg_right, disp_features_rc, COLOR_GRAY2RGB);
    //DebugHelper::drawFeatures(points, Scalar(255,0,0), disp_features_rc);
    //imshow("left_prev", disp_features_lp);
    //imshow("right_prev", disp_features_rp);
    //imshow("left_curr", disp_features_lc);
    //imshow("right_curr", disp_features_rc);
    // end debug

    tracker.track(img_left, img_right);
    //track::TrackerHelper::printTrackerStats(tracker);

#ifdef TSUKUBA
    Rt_prev = Rt_curr.clone();
    core::FormatHelper::ReadNextRtMatrix(groundtruth_file, Rt_curr);
    // read next depth GT data
    fs.open(depth_folder + depth_filelist[i/2], cv::FileStorage::READ);
    depth_prev = depth_curr;
    fs["depth"] >> depth_curr;
    cv::Mat Rt_2frames = Rt_prev.inv() * Rt_curr;
    int bad_num = track::EvalHelper::CountBadTracks(tracker, cam_params, Rt_2frames, depth_prev);
    double p = 100.0 * static_cast<double>(bad_num) / tracker.countActiveTracks();
    printf("Number of bad tracks = %d / %d = %f %%\n", bad_num, tracker.countActiveTracks(), p);
#endif

    cv::cvtColor(img_left_prev, disp_left_prev, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_right_prev, disp_right_prev, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_left, disp_left_curr, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_right, disp_right_curr, cv::COLOR_GRAY2RGB);
    //drawMatches(tracker, disp_left_prev, disp_right_prev, disp_left_curr, disp_right_curr);
    draw_flow(tracker, disp_left_prev, disp_right_prev);
    cv::waitKey(50);
  }
}

int main() {
  //std::string imagelistfn("/home/kivan/Projects/cv-stereo/config_files/kitti_07_lst.xml");
  //std::string source_folder("/home/kivan/Projects/datasets/KITTI/sequences_gray/07/");
  std::string imagelistfn("/home/kivan/Projects/cv-stereo/config_files/tsukuba_fluorescent_crop_lst.xml");
  std::string source_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/illumination/fluorescent/");
  std::vector<std::string> imagelist;

  bool ok = core::FormatHelper::readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty()) {
    std::cout << "can not open " << imagelistfn << " or the string list is empty\n";
    return -1;
  }
  run_tracker(imagelist, source_folder);

  return 0;
}
