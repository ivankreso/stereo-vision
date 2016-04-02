// implementation of tracking using KLT method from opencv

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "../../../core/image.h"
#include "../../../core/format_helper.h"
#include "../../../core/math_helper.h"
#include "../../base/helper_opencv.h"
#include "../../detector/feature_detector_harris_cv.h"
#include "../../mono/tracker_base.h"
#include "../../mono/tracker_opencv.h"
#include "../../mono/tracker_birch.h"
#include "../../mono/tracker_bfm.h"
#include "../../refiner/feature_refiner_klt.h"
#include "../../stereo/stereo_tracker_bfm.h"
#include "../../base/helper_opencv.h"

// matcher params
#define MIN_NCC                   0.9         // 0.8 - 0.9,     iva - 0.7
#define NCC_PATCH_SIZE            15          // 11 (Nister),  21(Irschara), 15 - refiner
#define MAX_FEATURES              5000       // 1000 - 2000, iva - 20000
#define SEARCH_WINDOW             140            // KITTI - 180, BB2 - 120, TSUKUBA - 140, iva - 300

//#define MAX_HAMM                  80          // FREAK - 60
//#define OUTLIER_TRACK_EPS         2.5 // 2.0

// detector params
#define HARRIS_BLOCK_SIZE         3     // 3, irchara uses 5 but more matches with 3
#define HARRIS_FILTER_SIZE        3     // 3, or 1
#define HARRIS_K                  0.04        // Nister - 0.06
// TODO:
#define HARRIS_THR                0.0000001         // 0.000001, iva - 0.0000001
//this depends on feature destcriptor used
// NCC_PATCH_SIZE+1/2 for NCC - +1 because of gradient image in FeatureRefinerKLT, 67 for FREAK
#define HARRIS_MARGIN             ((NCC_PATCH_SIZE+1) / 2)
#define HORIZ_BINS                10
#define VERTI_BINS                10
#define FEATURES_PER_BIN          20    // 20, try 15 for BB

#define SW_XL        1400
#define SW_XR        0
#define SW_YU        1
#define SW_YD        1

#define DRAW_ERROR_THR 1000.0
#define DEPTH_ERR_THR 1000.0

#define DRAW_POINT_MAX_DIST 2.0

std::vector<double> depth_triang, depth_triang_subpix, depth_gt;



void drawFlow(track::TrackerBase& tracker, cv::Mat& img)
{
  //double font_size = 0.3;
  cv::Point pt1, pt2;
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);
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

double triangulateDepth(core::Point& left, core::Point& right, double f, double baseline)
{
  double disp = left.x_ - right.x_;
  if(disp <= 0.0) {
    std::cout << "[EvalHelper]: zero/negative disparity: " << disp << "\n";
    return -1000.0;
  }

  double depth = f * baseline / disp;
  return depth;
}

void getDepthErrors(track::TrackerBase& tracker, std::vector<core::Point>& subpixel_right,
                     const double (&cam_params)[5], const cv::Mat& depth_mat,
                     std::vector<double>& depth_errors, std::vector<double>& depth_errors_subpix,
                     double& mae_depth, double& mae_depth_subpixel)
{
  depth_triang.assign(tracker.countFeatures(), -1.0);
  depth_triang_subpix.assign(tracker.countFeatures(), -1.0);
  depth_gt.assign(tracker.countFeatures(), -1.0);

  depth_errors.assign(tracker.countFeatures(), -1.0);
  depth_errors_subpix.assign(tracker.countFeatures(), -1.0);

  std::ofstream outfile_err("depth_errors.txt");
  std::ofstream outfile_error_subpix("depth_errors_subpixel.txt");
  double f = cam_params[0];
  double baseline = cam_params[4];
  double error_sum = 0.0;
  double error_sum_subpix = 0.0;
  int track_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo feat = tracker.feature(i);
    if(feat.age_ <= 0) continue;
    double depth = triangulateDepth(feat.prev_, feat.curr_, f, baseline);
    double depth_subpix = triangulateDepth(feat.prev_, subpixel_right[i], f, baseline);
    if(depth < 0.0 || depth_subpix < 0.0) {
      std::cout << "Negative triangulation!\n";
      tracker.removeTrack(i);
      continue;
    }
    depth_triang[i] = depth;
    depth_triang_subpix[i] = depth_subpix;
    // coords in left image
    double lx = feat.prev_.x_;
    double ly = feat.prev_.y_;
    int row = (int)std::round(ly);
    int col = (int)std::round(lx);
    double depth_error = std::abs(depth - depth_mat.at<float>(row, col));
    depth_gt[i] = depth_mat.at<float>(row, col);
    depth_errors[i] = depth_error;

    outfile_err << i << " " << depth_error << "\n";
    double depth_error_subpix = std::abs(depth_subpix - depth_mat.at<float>(row, col));
    depth_errors_subpix[i] = depth_error_subpix;
    outfile_error_subpix << i << " " << depth_error_subpix << "\n";
    if(depth_error_subpix > 1000.0) {
      std::cout << i << " - to big depth error:\nDepth error = " << depth_error << "\nDepth error (subpixel) = "
                << depth_error_subpix << "\n";
    }
    //if(depth_error > DEPTH_ERR_THR) {
    //  std::cout << "[EvalHelper]: warning - skippking big depth error!\n";
    //  continue;
    //}
    error_sum += depth_error;
    error_sum_subpix += depth_error_subpix;
    track_cnt++;
  }
  mae_depth = error_sum / track_cnt;
  mae_depth_subpixel = error_sum_subpix / track_cnt;
}

void findSubpixelFeatures(const core::Image& img_left, const core::Image& img_right, track::TrackerBase& tracker,
                          track::FeatureRefinerBase& refiner, std::vector<core::Point>& subpixel_feats)
{
  std::map<int, core::Point> init_refs, refined_right;

  subpixel_feats.resize(tracker.countFeatures());
  for(int i = 0; i < tracker.countFeatures(); i++)
  {
    track::FeatureInfo feat = tracker.feature(i);

    if(feat.age_ >= 1) {
      init_refs.insert(std::pair<int,core::Point>(i, feat.prev_));
      refined_right.insert(std::pair<int,core::Point>(i, feat.curr_));
    }
  }

  core::ImageSetExact imgset_left, imgset_right;
  imgset_left.compute(img_left);
  imgset_right.compute(img_right);

  refiner.addFeatures(imgset_left, init_refs);
  // refine the points in right frame
  refiner.refineFeatures(imgset_right, refined_right);
  for(auto pt : refined_right) {
    track::FeatureData fdata = refiner.getFeature(pt.first);
    core::Point rpt = fdata.pt();
    auto status = fdata.status_;
    if(status == track::FeatureData::OK) {
      subpixel_feats[pt.first] = rpt;
      //residue[pt.first] = std::make_tuple(fdata.first_residue_, fdata.residue_);
    }
    else
      tracker.removeTrack(pt.first);
  }
}


void drawTracks(const cv::Mat& img1, const cv::Mat& img2, track::TrackerBase& tracker,
                track::FeatureRefinerBase& refiner, const std::vector<core::Point>& subpixel_feats,
                const std::vector<double>& depth_errors, const std::vector<double>& depth_errors_subpix)
{
  //double font_size = 0.4; // 0.3
  cv::Point2f pt1, pt2, pt2subpix;
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);
  cv::Scalar color_ref(0,255,0);
  cv::Mat disp1, disp2, disp2_subpix, disp_all;
  cv::cvtColor(img1, disp_all, cv::COLOR_GRAY2RGB);
  uint32_t worse_cnt = 0;
  uint32_t active_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    cv::cvtColor(img1, disp1, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img2, disp2, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img2, disp2_subpix, cv::COLOR_GRAY2RGB);
    track::FeatureInfo feat = tracker.feature(i);
    //if(i != 2088) continue;

    if(feat.age_ > 0) {
      active_cnt++;
      //if(depth_errors[i] > depth_errors_subpix[i]) continue;
      double dist2d = core::MathHelper::getDist2D(feat.curr_, subpixel_feats[i]);
      if(dist2d <= DRAW_POINT_MAX_DIST) continue;
      if(depth_errors[i] < depth_errors_subpix[i]) {
        worse_cnt++;
        //continue;
      }

      std::cout << "---------------------\nDepth error tracker = " << depth_errors[i] << "\nDepth error subpix = "
                << depth_errors_subpix[i] << "\n";

      std::cout << "Depth tracker = " << depth_triang[i] << "\n";
      std::cout << "Depth subpixel = " << depth_triang_subpix[i] << "\n";
      std::cout << "Depth GT = " << depth_gt[i] << "\n";
      std::cout << "2D distance = " << dist2d << "\n";
      std::cout << "Got worse with subpixel = " << worse_cnt << " / " << active_cnt << "\n";
      pt1.x = feat.prev_.x_;
      pt1.y = feat.prev_.y_;
      pt2.x = feat.curr_.x_;
      pt2.y = feat.curr_.y_;
      pt2subpix.x = subpixel_feats[i].x_;
      pt2subpix.y = subpixel_feats[i].y_;
      cv::line(disp1, pt1, pt2, color_prev, 2, 8);
      cv::circle(disp1, pt1, 2, color_prev, -1, 8);
      cv::circle(disp1, pt2, 2, color_curr, -1, 8);
      cv::line(disp_all, pt1, pt2, color_prev, 0, 8);
      cv::circle(disp_all, pt1, 2, color_prev, -1, 8);
      cv::circle(disp_all, pt2, 2, color_curr, -1, 8);
      cv::circle(disp1, pt2subpix, 2, color_ref, -1, 8);
      cv::circle(disp_all, pt2subpix, 2, color_ref, -1, 8);
      std::cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << "\n";
      //cv::putText(img_lc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

      cv::circle(disp2, pt2, 2, color_curr, -1, 8);
      pt2subpix.x = subpixel_feats[i].x_;
      pt2subpix.y = subpixel_feats[i].y_;
      cv::circle(disp2_subpix, pt2subpix, 2, color_ref, -1, 8);
      std::cout << pt1.x << ", " << pt1.y << " --> " << pt2subpix.x << ", " << pt2subpix.y << "\n";
      //cv::putText(img_rc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

      double img_scale_sz = 1.5;
      cv::Mat disp1_scaled, disp2_scaled, disp2_subpix_scaled;
      cv::resize(disp1, disp1_scaled, cv::Size(), img_scale_sz, img_scale_sz);
      cv::resize(disp2, disp2_scaled, cv::Size(), img_scale_sz, img_scale_sz);
      cv::resize(disp2_subpix, disp2_subpix_scaled, cv::Size(), img_scale_sz, img_scale_sz);
      cv::imshow("first_image", disp1_scaled);
      cv::imshow("second_image", disp2_scaled);
      cv::imshow("second_image_subpixel", disp2_subpix_scaled);

      double scale_sz = 13.0;
      track::FeatureData r_data = refiner.getFeature(i);
      track::FeatureData t_data = tracker.getFeatureData(i);
      cv::Mat r_patch_warped, r_patch_ref, track_patch_ref, track_patch;
      track::HelperOpencv::FloatImageToMat(r_data.ref_, r_patch_ref);
      track::HelperOpencv::FloatImageToMat(r_data.warped_, r_patch_warped);
      cv::resize(r_patch_ref, r_patch_ref, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
      cv::resize(r_patch_warped, r_patch_warped, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
      cv::resize(t_data.patch_prev_, track_patch_ref, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
      cv::resize(t_data.patch_curr_, track_patch, cv::Size(), scale_sz, scale_sz, cv::INTER_NEAREST);
      cv::imshow("refiner_ref", r_patch_ref);
      cv::imshow("refiner_warped", r_patch_warped);
      cv::imshow("tracker_ref", track_patch_ref);
      cv::imshow("tracker_next", track_patch);
      int key = -1;
      while(key != 27 && key != 1048603) {
        key = cv::waitKey(0);
      }
    }
  }
  //cv::Mat disp_all_scaled;
  //cv::resize(disp_all, disp_all_scaled, cv::Size(), 3.0, 3.0);
  //cv::imshow("flow_all", disp_all_scaled);
  //cv::waitKey();

  //vector<int> compression_params;
  //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  //compression_params.push_back(9);
  //cv::imwrite("tracks_left.png", img_lc, compression_params);
  //cv::imwrite("tracks_right.png", img_rc, compression_params);
  //cv::waitKey(0);
}


void debug(const cv::Mat& img_left, const cv::Mat& img_right, track::TrackerBase& tracker,
           track::FeatureRefinerBase& refiner, const std::vector<core::Point>& subpixel_right,
           std::vector<double>& depth_errors, std::vector<double>& depth_errors_subpix)
{
  drawTracks(img_left, img_right, tracker, refiner, subpixel_right, depth_errors, depth_errors_subpix);
}

void runEvaluation(std::string source_folder, std::vector<std::string>& imglist)
{
  //std::ifstream groundtruth_file("/home/kivan/Projects/datasets/KITTI/poses/07.txt");
  std::string dispfn("/home/kivan/Projects/cv-stereo/config_files/truskuba_disp_lst.xml");
  std::string cparams_file("/home/kivan/Projects/cv-stereo/config_files/camera_params_tsukuba.txt");
  std::string depth_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/groundtruth/depth_maps/left/");
  std::string depthfn("/home/kivan/Projects/cv-stereo/config_files/tsukuba_depth_lst.xml");
  std::vector<std::string> depth_filelist;
  core::FormatHelper::readStringList(depthfn, depth_filelist);
  std::ofstream outfile_mae("depth_mae.txt");
  std::ofstream outfile_mae_subpixel("depth_mae_subpixel.txt");

  double cam_params[5];
  core::FormatHelper::readCameraParams(cparams_file, cam_params);
  cv::Mat depth_mat;

  size_t start_frame = 129;
  cv::Mat cvimg_left, cvimg_right;

  //Mat disp_left_prev, disp_right_prev, disp_left_ref, disp_right_ref;
  //Mat disp_features_lp, disp_features_rp, disp_features_lc, disp_features_rc;
  core::Image img_left, img_right;
  std::vector<core::Point> subpixel_right;

  track::FeatureDetectorHarrisCV detector(HARRIS_BLOCK_SIZE, HARRIS_FILTER_SIZE, HARRIS_K, HARRIS_THR, HARRIS_MARGIN);
  track::TrackerBFM tracker(detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SW_XL, SW_XR, SW_YU, SW_YD);

  for(size_t i = start_frame*2; i < imglist.size(); i+=2) {
    std::cout << "::::::::: " << i/2 << ". frame :::::::::\n";
    cvimg_left = cv::imread(source_folder + imglist[i], CV_LOAD_IMAGE_GRAYSCALE);
    cvimg_right = cv::imread(source_folder + imglist[i+1], CV_LOAD_IMAGE_GRAYSCALE);
    track::HelperOpencv::MatToImage(cvimg_left, img_left);
    track::HelperOpencv::MatToImage(cvimg_right, img_right);

    std::string depth_gt_path = depth_folder + depth_filelist[i/2];
    cv::FileStorage fs(depth_gt_path, cv::FileStorage::READ);
    std::cout << depth_gt_path << "\n";
    fs["depth"] >> depth_mat;

    //track::TrackerBFM tracker(detector, MAX_FEATURES, MIN_NCC, NCC_PATCH_SIZE, SW_XL, SW_XR, SW_YU, SW_YD);
    tracker.init(img_left);
    tracker.track(img_right);
    tracker.printStats();
    track::FeatureRefinerKLT refiner;
    findSubpixelFeatures(img_left, img_right, tracker, refiner, subpixel_right);
    double mae, mae_subpixel;
    std::vector<double> depth_errors, depth_errors_subpix;
    getDepthErrors(tracker, subpixel_right, cam_params, depth_mat, depth_errors, depth_errors_subpix,
                    mae, mae_subpixel);
    std::cout << "MAE: " << mae << "\n";
    std::cout << "MAE subpixel: " << mae_subpixel << "\n";
    outfile_mae << mae << "\n";
    outfile_mae_subpixel << mae_subpixel << "\n";
    //return;
    debug(cvimg_left, cvimg_right, tracker, refiner, subpixel_right, depth_errors, depth_errors_subpix);
    //cv::cvtColor(cvimg_prev, disp, COLOR_GRAY2RGB);
    //drawFlow(tracker, disp);
    cv::waitKey(0);

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



    //for(int j = 0; j < tracker.countFeatures(); j++) {
    //  track::FeatureInfo feat = tracker.feature(j);
    //  if(feat.age_ > 0) {
    //    std::cout << feat.prev_ << "\n";
    //    std::cout << feat.curr_ << "\n";
    //    out << feat.prev_.x_ << " " << feat.prev_.y_ << " " << feat.curr_.x_ << " " << feat.curr_.y_ << "\n";
    //  }
    //}

  }
}

inline static bool readStringList(const std::string& filename, std::vector<std::string>& strlist)
{
  strlist.resize(0);
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if(!fs.isOpened())
    return false;
  cv::FileNode n = fs.getFirstTopLevelNode();
  if(n.type() != cv::FileNode::SEQ)
    return false;
  cv::FileNodeIterator it = n.begin(), it_end = n.end();
  for(; it != it_end; ++it)
    strlist.push_back((std::string)*it);
  return true;
}


int main()
{
  std::string source_folder("/home/kivan/Projects/datasets/Tsukuba/NewTsukubaStereoDataset/illumination/fluorescent/");
  std::string imagelistfn("/home/kivan/Projects/cv-stereo/config_files/tsukuba_fluorescent_crop_lst.xml");
  std::vector<std::string> imagelist;
  std::vector<std::string> disp_imagelist;

  bool ok = readStringList(imagelistfn, imagelist);
  //ok = readStringList(dispfn, disp_imagelist);
  if(!ok || imagelist.empty())
  {
    std::cout << "can not open " << imagelistfn << " or the string list is empty\n";
    return -1;
  }
  runEvaluation(source_folder, imagelist);

  return 0;
}
