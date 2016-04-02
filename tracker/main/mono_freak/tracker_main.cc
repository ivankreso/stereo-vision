// implementation of tracking using KLT method from opencv

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
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
#include "../../mono/tracker_bfm_cv.h"
#include "../../detector/feature_detector_harris_cv.h"
#include "../../detector/feature_detector_brisk.h"
#include "../../detector/feature_detector_harris_freak.h"

// matcher params
#define MAX_FEATURES              4096        // Kanade - 4096
#define SEARCH_WINDOW             200            // KITTI - 180, BB2 - 120, TSUKUBA - 100, iva - 300

//#define MAX_HAMM                  80          // FREAK - 60
//#define OUTLIER_TRACK_EPS         2.5 // 2.0

// detector params
#define HARRIS_BLOCK_SIZE         3     // 3, irchara uses 5 but more matches with 3
#define HARRIS_FILTER_SIZE        3     // 3, or 1
#define HARRIS_K                  0.06        // Nister - 0.06
// TODO:
#define HARRIS_THR                0.0000001         // 0.000001, iva - 0.0000001
//this depends on feature destcriptor used
#define HARRIS_MARGIN             10     // NCC_PATCH_SIZE/2 for NCC, 67 for FREAK
#define HORIZ_BINS                10
#define VERTI_BINS                10
#define FEATURES_PER_BIN          20    // 20, try 15 for BB

void drawFlow(track::TrackerBase& tracker, cv::Mat& img)
{
  double font_size = 0.3;
  int color = 255;
  cv::Point pt1, pt2;
  Scalar color_curr(0,0,255);
  Scalar color_prev(255,0,0);
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat = tracker.feature(i);
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
  cv::imshow("mono_track", img);
}

void runTracker(string source_folder, vector<string>& imglist)
{
  int start_index = 0;
  cv::Mat cvimg = imread(source_folder + imglist[start_index], CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cvimg_prev, disp;

  //TrackerBFM tracker(detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SEARCH_WINDOW);
  //track::FeatureDetectorBRISK detector(30, 3, 1.0);
  //track::FeatureDetectorBRISK detector(10, 3, 1.0);
  track::FeatureDetectorHarrisCV detector_base(HARRIS_BLOCK_SIZE, HARRIS_FILTER_SIZE, HARRIS_K,
                                               HARRIS_THR, HARRIS_MARGIN);
  //cv::FREAK extractor(true, false);
  cv::FREAK extractor(true, false, 22.0f, 0);
  track::FeatureDetectorHarrisFREAK detector(detector_base, extractor);
  track::TrackerBFMcv tracker(detector, MAX_FEATURES, SEARCH_WINDOW, 50);
  //track::TrackerBFMcv tracker(detector, MAX_FEATURES, SEARCH_WINDOW, 50);

  tracker.init(cvimg);

  //for(size_t i = start_index+1; i < imglist.size(); i++) {
  for(size_t i = start_index+2; i < imglist.size(); i+=2) {
    cvimg_prev = cvimg;
    cvimg = imread(source_folder + imglist[i], CV_LOAD_IMAGE_GRAYSCALE);

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

    cv::cvtColor(cvimg_prev, disp, COLOR_GRAY2RGB);
    tracker.track(cvimg);
    drawFlow(tracker, disp);

    //ofstream out("tracks_" + std::to_string(i) + ".txt");
    //for(int j = 0; j < tracker.countFeatures(); j++) {
    //  track::FeatureInfo feat = tracker.feature(j);
    //  if(feat.age_ > 0) {
    //    cout << feat.prev_ << "\n";
    //    cout << feat.curr_ << "\n";
    //    out << feat.prev_.x_ << " " << feat.prev_.y_ << " " << feat.curr_.x_ << " " << feat.curr_.y_ << "\n";
    //  }
    //}
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
  string source_folder("/home/kivan/Projects/datasets/KITTI/sequences_gray/07/");
  string imagelistfn("/home/kivan/Projects/cv-stereo/config_files/kitti_07_lst.xml");
  vector<string> imagelist;
  vector<string> disp_imagelist;

  bool ok = readStringList(imagelistfn, imagelist);
  //ok = readStringList(dispfn, disp_imagelist);
  if(!ok || imagelist.empty())
  {
    cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    return -1;
  }
  runTracker(source_folder, imagelist);

  return 0;
}
