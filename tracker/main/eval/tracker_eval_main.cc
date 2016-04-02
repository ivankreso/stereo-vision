// implementation of tracking using KLT method from opencv

#include <iostream>
#include <vector>
#include <string>
using namespace std;

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/core/operations.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

#include "../../../core/image.h"
#include "../../mono/tracker_base.h"
#include "../../mono/tracker_opencv.h"
#include "../../mono/tracker_birch.h"
#include "../../mono/tracker_bfm.h"
#include "../../base/helper_opencv.h"
#include "../../stereo/stereo_tracker_base.h"
#include "../../stereo/stereo_tracker.h"
#include "../../stereo/stereo_tracker_bfm.h"
#include "../../stereo/stereo_tracker_refiner.h"
#include "../../detector/feature_detector_harris_cv.h"
#include "../../detector/feature_detector_gftt_cv.h"
#include "../../refiner/feature_refiner_base.h"
#include "../../refiner/feature_refiner_klt.h"
using namespace track;
using namespace core;


// matcher params
#define MIN_NCC                   0.9         // 0.8 - 0.9,     iva - 0.7
#define NCC_PATCH_SIZE            21          // 11 (Nister),  21(Irschara)
#define MAX_HAMM                  80          // FREAK - 60
#define MAX_FEATURES              5000        // tsukuba - 5000, iva - 20000
#define SEARCH_WINDOW             100            // KITTI - 180, BB2 - 120, TSUKUBA - 100, iva - 300
//#define OUTLIER_TRACK_EPS         2.5 // 2.0

// detector params
#define HARRIS_BLOCK_SIZE         3     // 3, irchara uses 5 but more matches with 3
#define HARRIS_FILTER_SIZE        3     // 3, or 1
#define HARRIS_K                  0.04        // Nister - 0.06
// TODO:
#define HARRIS_THR                0.0000001         // 0.000001, iva - 0.0000001
//this depends on feature destcriptor used
#define HARRIS_MARGIN             (NCC_PATCH_SIZE / 2)     // NCC_PATCH_SIZE/2 for NCC, 67 for FREAK
#define HORIZ_BINS                10
#define VERTI_BINS                10
#define FEATURES_PER_BIN          20    // 20, try 15 for BB

void drawMatches(StereoTrackerBase& tracker, cv::Mat& img_lp, cv::Mat& img_rp, Mat& img_lc, Mat& img_rc)
{
  cv::Point2f pt1, pt2;
  Scalar color(0,0,0);
  for(int i = 0; i < tracker.countFeatures(); i++) {
    FeatureInfo feat_left = tracker.featureLeft(i);
    FeatureInfo feat_right = tracker.featureRight(i);
    //cout << feat_left.status_ << endl;
    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      Mat img_lp_orig = img_lp.clone();
      Mat img_rp_orig = img_rp.clone();
      Mat img_lc_orig = img_lc.clone();
      Mat img_rc_orig = img_rc.clone();
      color = Scalar(255,0,0);
      pt1.x = feat_left.prev_.x_;
      pt1.y = feat_left.prev_.y_;
      pt2.x = feat_left.curr_.x_;
      pt2.y = feat_left.curr_.y_;
      //cv::line(img_lc, pt1, pt2, color, 2, 8);
      cv::circle(img_lp, pt1, 4, color, -1, 8);
      cv::circle(img_lc, pt2, 4, color, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;

      pt1.x = feat_right.prev_.x_;
      pt1.y = feat_right.prev_.y_;
      pt2.x = feat_right.curr_.x_;
      pt2.y = feat_right.curr_.y_;
      //cv::line(img_lc, pt1, pt2, color, 2, 8);
      cv::circle(img_rp, pt1, 4, color, -1, 8);
      cv::circle(img_rc, pt2, 4, color, -1, 8);
      cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      imshow("left_prev_track", img_lp);
      imshow("left_curr_track", img_lc);
      imshow("right_prev_track", img_rp);
      imshow("right_curr_track", img_rc);
      waitKey(0);
      img_lp = img_lp_orig;
      img_lc = img_lc_orig;
      img_rp = img_rp_orig;
      img_rc = img_rc_orig;
    }
  }
}

void drawStereoFlow(StereoTrackerBase& tracker, cv::Mat& img_l, cv::Mat& img_r,
                    std::string left_name, std::string right_name)
{
  double font_size = 0.3;
  int color = 255;
  cv::Point pt1, pt2;
  Scalar color_curr(0,0,255);
  Scalar color_prev(255,0,0);
  for(int i = 0; i < tracker.countFeatures(); i++) {
    FeatureInfo feat_left = tracker.featureLeft(i);
    FeatureInfo feat_right = tracker.featureRight(i);
    //cout << feat_left.status_ << endl;
    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      pt1.x = feat_left.prev_.x_;
      pt1.y = feat_left.prev_.y_;
      pt2.x = feat_left.curr_.x_;
      pt2.y = feat_left.curr_.y_;
      cv::line(img_l, pt1, pt2, color_prev, 2, 8);
      cv::circle(img_l, pt1, 2, color_prev, -1, 8);
      cv::circle(img_l, pt2, 2, color_curr, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      //cv::putText(img_lc, to_string(i), pt1, FONT_HERSHEY_SIMPLEX, font_size, Scalar::all(color), 1, 8);

      pt1.x = feat_right.prev_.x_;
      pt1.y = feat_right.prev_.y_;
      pt2.x = feat_right.curr_.x_;
      pt2.y = feat_right.curr_.y_;
      cv::line(img_r, pt1, pt2, color_prev, 2, 8);
      cv::circle(img_r, pt1, 2, color_prev, -1, 8);
      cv::circle(img_r, pt2, 2, color_curr, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      //cv::putText(img_rc, to_string(i), pt1, FONT_HERSHEY_SIMPLEX, font_size, Scalar::all(color), 1, 8);
    }
  }
  imshow(left_name, img_l);
  imshow(right_name, img_r);
}

void drawFlow(TrackerBase& tracker, cv::Mat& img)
{
  double font_size = 0.3;
  int color = 255;
  cv::Point pt1, pt2;
  Scalar color_curr(0,0,255);
  Scalar color_prev(255,0,0);
  for(int i = 0; i < tracker.countFeatures(); i++) {
    FeatureInfo feat = tracker.feature(i);
    //cout << feat_left.status_ << endl;
    if(feat.age_ > 0) {
      pt1.x = feat.prev_.x_;
      pt1.y = feat.prev_.y_;
      pt2.x = feat.curr_.x_;
      pt2.y = feat.curr_.y_;
      cv::line(img, pt1, pt2, color_prev, 2, 8);
      cv::circle(img, pt1, 2, color_prev, -1, 8);
      cv::circle(img, pt2, 2, color_curr, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      //cv::putText(img, to_string(i), pt1, FONT_HERSHEY_SIMPLEX, font_size, Scalar::all(color), 1, 8);
    }
  }
  imshow("mono_track", img);
}

void runTracker(string source_folder, vector<string>& imglist, string src_disp, vector<string>& disp_imglist)
{
  int start_index = 0;
  Mat cvimg_left = imread(source_folder + imglist[start_index], CV_LOAD_IMAGE_GRAYSCALE);
  Mat cvimg_right = imread(source_folder + imglist[start_index+1], CV_LOAD_IMAGE_GRAYSCALE);
  //Mat disp_gt = imread(src_disp + disp_imglist[start_index], CV_LOAD_IMAGE_GRAYSCALE);
  //cout << disp_gt << "\n\n";

  Mat cvimg_left_prev, cvimg_right_prev;
  //equalizeHist(cvimg_left, cvimg_left);
  //equalizeHist(cvimg_right, cvimg_right);

  //HelperOpencv::moveImageToMat(img, img_mat);
  Mat disp_left_prev, disp_right_prev, disp_left_ref, disp_right_ref;
  cv::Mat disp_mono;
  Mat disp_features_lp, disp_features_rp, disp_features_lc, disp_features_rc;
  Mat img_rgb;
  //cvtColor(cvimg_left, disp_left_track, COLOR_GRAY2RGB);
  //imshow("left_track", disp_left_track);
  //waitKey(0);

  core::Image img_left, img_right, img_left_prev, img_right_prev;
  HelperOpencv::MatToImage(cvimg_left, img_left);
  HelperOpencv::MatToImage(cvimg_right, img_right);

  FeatureDetectorHarrisCV detector(HARRIS_BLOCK_SIZE, HARRIS_FILTER_SIZE, HARRIS_K, HARRIS_THR, HARRIS_MARGIN);
  StereoTrackerBFM tracker(&detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SEARCH_WINDOW);

  StereoTrackerBFM tracker_basic(&detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SEARCH_WINDOW);
  FeatureRefinerKLT refiner;
  StereoTrackerRefiner tracker_refiner(&tracker_basic, &refiner, tracker_basic.countFeatures());

  TrackerBFM tracker_mono(detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SEARCH_WINDOW);

  tracker_mono.init(img_left);
  tracker.init(img_left, img_right);
  tracker_refiner.init(img_left, img_right);

  for(size_t i = start_index+2; i < imglist.size(); i+=2) {
    cvimg_left_prev = cvimg_left.clone();
    cvimg_right_prev = cvimg_right.clone();
    img_left_prev = img_left;
    img_right_prev = img_right;
    cvimg_left = imread(source_folder + imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
    cvimg_right = imread(source_folder + imglist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    //equalizeHist(cvimg_left, cvimg_left);
    //equalizeHist(cvimg_right, cvimg_right);

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

    cvtColor(cvimg_left_prev, disp_left_prev, COLOR_GRAY2RGB);
    cvtColor(cvimg_left_prev, disp_mono, COLOR_GRAY2RGB);
    cvtColor(cvimg_right_prev, disp_right_prev, COLOR_GRAY2RGB);
    cvtColor(cvimg_left_prev, disp_left_ref, COLOR_GRAY2RGB);
    cvtColor(cvimg_right_prev, disp_right_ref, COLOR_GRAY2RGB);
    HelperOpencv::MatToImage(cvimg_left, img_left);
    HelperOpencv::MatToImage(cvimg_right, img_right);

    tracker_mono.track(img_left);
    tracker_mono.printStats();
    tracker.track(img_left, img_right);
    tracker.printStats();
    tracker_refiner.track(img_left, img_right);
    tracker_refiner.printStats();

    //drawMatches(tracker, disp_left_prev, disp_right_prev, disp_left_curr, disp_right_curr);
    drawStereoFlow(tracker, disp_left_prev, disp_right_prev, "left", "right");
    drawStereoFlow(tracker_refiner, disp_left_ref, disp_right_ref, "left_refined", "right_refined");
    drawFlow(tracker_mono, disp_mono);

    waitKey(10);
  }
}


inline static bool readStringList(const string& filename, vector<string>& strlist)
{
  strlist.resize(0);
  FileStorage fs(filename, FileStorage::READ);
  if(!fs.isOpened())
    return false;
  FileNode n = fs.getFirstTopLevelNode();
  if(n.type() != FileNode::SEQ)
    return false;
  FileNodeIterator it = n.begin(), it_end = n.end();
  for(; it != it_end; ++it)
    strlist.push_back((string)*it);
  return true;
}


int main()
{
  string imagelistfn("/home/kivan/Projects/vista-stereo/config_files/tsukuba_fluorescent_lst.xml");
  string source_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/illumination/fluorescent/");
  string src_disp("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/disparity_maps/");
  string dispfn("/home/kivan/Projects/vista-stereo/config_files/truskuba_disp_lst.xml");
  //string source_folder("/home/kreso/Projects/datasets/KITTI/sequences_gray/");
  vector<string> imagelist;
  vector<string> disp_imagelist;

  bool ok = readStringList(imagelistfn, imagelist);
  ok = readStringList(dispfn, disp_imagelist);
  if(!ok || imagelist.empty())
  {
    cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }
  runTracker(source_folder, imagelist, src_disp, disp_imagelist);

  return 0;
}
