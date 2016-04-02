#include "eval_helper.h"

#include <sstream>
#include <fstream>
#include <unordered_set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include "../../core/math_helper.h"

using namespace std;
using namespace cv;

namespace {

double triangulateDepth(core::Point& left, core::Point& right, double f, double baseline)
{
  double disp = left.x_ - right.x_;
  if(disp <= 0.0) {
    std::cout << "[EvalHelper]: zero/negative disparity: " << disp << "\n";
    //exit(-1);
  }

  double depth = f * baseline / disp;
  return depth;
}

//void Get3DPoint(const core::Point& pt, const double (&cam_params)[5], const cv::Mat& depth,
//                  Eigen::Vector4d& pt3d)
//{
//  double f = cam_params[0];
//  double cx = cam_params[2];
//  double cy = cam_params[3];
//  int x = (int)std::round(pt.x_);
//  int y = (int)std::round(pt.y_);
//  pt3d(2) = depth.at<float>(y,x);
//  pt3d(0) = (x - cx) * pt3d[2] / f;
//  pt3d(1) = (y - cy) * pt3d[2] / f;
//  //pt3d(1) = (y - cy) * b / disp;
//  //pt3d(2) = f * b / disp;
//  pt3d(3) = 1.0;
//}

cv::Mat get_camera_matrix(const double* cam_params)
{
  cv::Mat C = cv::Mat::eye(3, 4, CV_64F);
  C.at<double>(0,0) = cam_params[0];
  C.at<double>(1,1) = cam_params[1];
  C.at<double>(0,2) = cam_params[2];
  C.at<double>(1,2) = cam_params[3];
  return C;
}

void point_to_cvmat(core::Point& pt, Mat& mat) {
  mat.at<double>(0) = pt.x_;
  mat.at<double>(1) = pt.y_;
  mat.at<double>(2) = 1.0;
}

void save_patch(const std::string& folder, const cv::Mat& patch, const std::string& filename)
{
  //cv::imwrite(folder + "/" + filename, patch, CV_IMWRITE_PXM_BINARY);
  cv::imwrite(folder + "/" + filename, patch);
}

}

namespace track {

#define DRAW_ERROR_THR 1000.0
#define DEPTH_ERR_THR 1000.0

bool EvalHelper::FilterOutliersWithGroundtruth(track::StereoTrackerBase& tracker, const double* cam_params,
                                               const cv::Mat& cvRt, const double error_thr)
{
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);
  double error_3d, left_reproj_error, right_reproj_error;

  std::vector<size_t> outliers;
  int active_tracks = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if(pt_left.age_ < 1) continue;

    active_tracks++;
    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          error_3d, left_reproj_error, right_reproj_error);

    if(left_reproj_error > error_thr || right_reproj_error > error_thr)
      outliers.push_back(i);
  }
  std::cout << "[EvalHelper::FilterOutliersWithGroundtruth]: number of errors above thr (" << error_thr
    << ") = " << outliers.size() << "\n";
  if(outliers.size() < 0.6 * active_tracks) {
    for(size_t i = 0; i < outliers.size(); i++)
      tracker.removeTrack(outliers[i]);
  }
  else return false;
  return true;
}

void EvalHelper::DrawPointPatch(const core::Point& pt, const cv::Mat& img, const int size,
                                size_t resolution, bool save_on_disk, const std::string filename) {
  assert(size % 2 == 1);
  int hsize = size / 2;
  int sx = std::max(0, int(std::round(pt.x_) - hsize));
  int sy = std::max(0, int(std::round(pt.y_) - hsize));
  int width = std::min(size, img.cols - sx);
  int height = std::min(size, img.rows - sy);
  cv::Rect rect(sx, sy, width, height);
  //std::cout << rect << "\n";
  cv::Mat patch = img(rect);
  cv::Mat scaled;
  cv::resize(patch, scaled, cv::Size(resolution, resolution), 0, 0, cv::INTER_NEAREST);
  //cv::resize
  cv::imshow(filename, scaled);
  if (save_on_disk)
    cv::imwrite(filename, scaled);
}

void EvalHelper::DrawTrackPatches(track::StereoTrackerBase& tracker, int track_num, int win_size,
                                  const cv::Mat& img_lp, const cv::Mat& img_rp,
                                  const cv::Mat& img_lc, const cv::Mat& img_rc) {
  track::FeatureInfo left = tracker.featureLeft(track_num);
  track::FeatureInfo right = tracker.featureRight(track_num);

  size_t resolution = 200;
  size_t x_pos = 1000;
  DrawPointPatch(left.prev_, img_lp, win_size, resolution, false, "LP_patch");
  DrawPointPatch(right.prev_, img_rp, win_size, resolution, false, "RP_patch");
  DrawPointPatch(left.curr_, img_lc, win_size, resolution, false, "LC_patch");
  DrawPointPatch(right.curr_, img_rc, win_size, resolution, false, "RC_patch");
  cv::moveWindow("LP_patch", x_pos, 1080-resolution-28);
  cv::moveWindow("RP_patch", x_pos + resolution+4, 1080-resolution-28);
  cv::moveWindow("LC_patch", x_pos + 2*(resolution+4), 1080-resolution-28);
  cv::moveWindow("RC_patch", x_pos + 3*(resolution+4), 1080-resolution-28);
}

void EvalHelper::DrawStereoTrack(track::StereoTrackerBase& tracker, int i, cv::Mat& img_lp, cv::Mat& img_rp,
                                 cv::Mat& img_lc, cv::Mat& img_rc, const int pt_size, cv::Scalar color)
{
  //double font_size = 0.4; // 0.3
  cv::Point pt1, pt2;
  //int pt_size = 5;
  //int pt_size = 0;
  track::FeatureInfo feat_left = tracker.featureLeft(i);
  track::FeatureInfo feat_right = tracker.featureRight(i);
  //cout << feat_left.status_ << endl;
  assert(feat_left.age_ > 0 && feat_right.age_ > 0);
  if(feat_left.age_ > 0 && feat_right.age_ > 0) {
    pt1.x = std::round(feat_left.prev_.x_);
    pt1.y = std::round(feat_left.prev_.y_);
    pt2.x = std::round(feat_left.curr_.x_);
    pt2.y = std::round(feat_left.curr_.y_);
    cv::circle(img_lp, pt1, pt_size, color, -1, 8);
    cv::circle(img_lc, pt2, pt_size, color, -1, 8);
    //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
    //cv::putText(img_lc, to_string(i), pt1, cv::FONT_HERS 0, 0,HEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

    pt1.x = std::round(feat_right.prev_.x_);
    pt1.y = std::round(feat_right.prev_.y_);
    pt2.x = std::round(feat_right.curr_.x_);
    pt2.y = std::round(feat_right.curr_.y_);
    cv::circle(img_rp, pt1, pt_size, color, -1, 8);
    cv::circle(img_rc, pt2, pt_size, color, -1, 8);
    //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
    //cv::putText(img_rc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);
  }

  //cv::imshow("left_prev_track", img_lp);
  //cv::imshow("right_prev_track", img_rp);
  //cv::imshow("left_curr_track", img_lc);
  //cv::imshow("right_curr_track", img_rc);
}

void EvalHelper::SaveTrackerEvaluation(
    track::StereoTrackerBase& tracker, const double* cam_params,
    const cv::Mat& cvRt, double max_error) {
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);
  double error_3d, left_reproj_error, right_reproj_error;

  std::ofstream response_file("errors_response.txt", std::ios_base::app);
  std::ofstream matching_file("errors_matching.txt", std::ios_base::app);
  std::ofstream disparity_file("errors_disparity.txt", std::ios_base::app);

  for (int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if (pt_left.age_ < 1) continue;

    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          error_3d, left_reproj_error, right_reproj_error);

    if ((left_reproj_error <= max_error && right_reproj_error <= max_error)) {
      track::TrackStats stats = tracker.GetTrackStats(i);
      response_file << stats.left_response_sum << " " << left_reproj_error << std::endl;
      response_file << stats.right_response_sum << " " << right_reproj_error << std::endl;
      matching_file << stats.matching_distance << " "
                    << left_reproj_error + right_reproj_error << std::endl;
      disparity_file << pt_left.prev_.x_ - pt_right.prev_.x_ << " "
                     << left_reproj_error + right_reproj_error << std::endl;
    }
  }
}

size_t EvalHelper::DrawTracksWithErrors(
    track::StereoTrackerBase& tracker, const double* cam_params, const cv::Mat& cvRt,
    const cv::Mat& img_lp, const cv::Mat& img_rp, const cv::Mat& img_lc, const cv::Mat& img_rc,
    double min_error) {
  cv::Mat lp_draw, rp_draw, lc_draw, rc_draw;
  const cv::Scalar color = cv::Scalar(0,0,255);
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);
  double error_3d, left_reproj_error, right_reproj_error;

  size_t outliers_cnt = 0;
  bool draw = false;
  for (int i = 0; i < tracker.countFeatures(); i++) {
    //if (i != 676) continue;
    //if (i != 219) continue;
    //if (i != 121) continue;
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if (pt_left.age_ < 1) continue;

    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          error_3d, left_reproj_error, right_reproj_error);

    if ((left_reproj_error > min_error || right_reproj_error > min_error)) {
      outliers_cnt++;
      if (!draw) continue;
    //if ((left_reproj_error < min_error && right_reproj_error < min_error)) {
    //if ((left_reproj_error > min_error || right_reproj_error > min_error)
        //&& (left_reproj_error < max_error || right_reproj_error < max_error)) {

    //if (pt_left.prev_.y_ != pt_right.prev_.y_ || pt_left.curr_.y_ != pt_right.curr_.y_) {
    //if (std::abs(pt_left.curr_.x_ - cam_params[2]) > img_lp.cols/3
    //    || std::abs(pt_right.curr_.x_ - cam_params[2]) > img_lp.cols/3) {


      //drawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, cv::Scalar(0,255,0));
      std::cout << "LP = " << pt_left.prev_ << "\nRP = " << pt_right.prev_
                << "\nLC = " << pt_left.curr_ << "\nRC = " << pt_right.curr_ << "\n";
      std::cout << "ID = " << i << "\n";
      std::cout << "LEFT error = " << left_reproj_error << "\n";
      std::cout << "RIGHT error = " << right_reproj_error << "\n\n";
      track::TrackStats stats = tracker.GetTrackStats(i);
      std::cout << "Left response sum = " << stats.left_response_sum << "\n";
      std::cout << "Right response sum = " << stats.right_response_sum << "\n";
      std::cout << "Matching distance sum = " << stats.matching_distance << "\n";

      //if (std::abs(pt_left.curr_.y_ - pt_right.curr_.y_) < 16) continue;

      cv::cvtColor(img_lp, lp_draw, cv::COLOR_GRAY2RGB);
      cv::cvtColor(img_rp, rp_draw, cv::COLOR_GRAY2RGB);
      cv::cvtColor(img_lc, lc_draw, cv::COLOR_GRAY2RGB);
      cv::cvtColor(img_rc, rc_draw, cv::COLOR_GRAY2RGB);
      DrawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, 0, color);
      DrawTrackPatches(tracker, i, 15, lp_draw, rp_draw, lc_draw, rc_draw);
      DrawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, 4, color);
      double s = 352.0 / lp_draw.rows;
      cv::resize(lp_draw, lp_draw, cv::Size(), s, s);
      cv::resize(rp_draw, rp_draw, cv::Size(), s, s);
      cv::resize(lc_draw, lc_draw, cv::Size(), s, s);
      cv::resize(rc_draw, rc_draw, cv::Size(), s, s);
      cv::imshow("RC", rc_draw);
      cv::imshow("RP", rp_draw);
      cv::imshow("LC", lc_draw);
      cv::imshow("LP", lp_draw);
      cv::moveWindow("RP", 1920-rp_draw.cols, 0);
      cv::moveWindow("RC", 1920-rp_draw.cols, 60+rc_draw.rows);
      cv::moveWindow("LP", 0, 0);
      cv::moveWindow("LC", 0, 60+rc_draw.rows);
      cv::waitKey(0);
    //}
    }
  }
  return outliers_cnt;
}

void EvalHelper::DrawTracksWithBigErrors(track::StereoTrackerBase& tracker, const double* cam_params,
                                         const cv::Mat& cvRt, const double min_error, const double max_error,
                                         const int num, const cv::Mat& img_lp, const cv::Mat& img_rp,
                                         const cv::Mat& img_lc, const cv::Mat& img_rc,
                                         const cv::Scalar& color, bool redraw_image,
                                         cv::Mat& lp_draw, cv::Mat& rp_draw, cv::Mat& lc_draw, cv::Mat& rc_draw)
{
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);
  double error_3d, left_reproj_error, right_reproj_error;

  std::vector<size_t> outliers;
  int active_tracks = 0;
  for (int i = 0; i < tracker.countFeatures(); i++) {
    if (num >= 0)
      i = num;
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if (pt_left.age_ < 1) continue;

    active_tracks++;
    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          error_3d, left_reproj_error, right_reproj_error);

    if ((left_reproj_error > min_error || right_reproj_error > min_error)
        && (left_reproj_error < max_error || right_reproj_error < max_error)) {
    //if (pt_left.prev_.y_ != pt_right.prev_.y_ || pt_left.curr_.y_ != pt_right.curr_.y_) {
    //if (std::abs(pt_left.curr_.x_ - cam_params[2]) > img_lp.cols/3
    //    || std::abs(pt_right.curr_.x_ - cam_params[2]) > img_lp.cols/3) {
      //cv::Mat lp_draw, rp_draw, lc_draw, rc_draw;
      if (redraw_image) {
        cv::cvtColor(img_lp, lp_draw, cv::COLOR_GRAY2RGB);
        cv::cvtColor(img_rp, rp_draw, cv::COLOR_GRAY2RGB);
        cv::cvtColor(img_lc, lc_draw, cv::COLOR_GRAY2RGB);
        cv::cvtColor(img_rc, rc_draw, cv::COLOR_GRAY2RGB);
      }
      DrawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, 0, color);
      DrawTrackPatches(tracker, i, 15, lp_draw, rp_draw, lc_draw, rc_draw);
      DrawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, 4, color);

      //drawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, cv::Scalar(0,255,0));
      std::cout << pt_left.prev_ << "\n" << pt_right.prev_ << "\n" << pt_left.curr_
                << "\n" << pt_right.curr_ << "\n";
      std::cout << i << "\n";
      std::cout << "LEFT error = " << left_reproj_error << "\n";
      std::cout << "RIGHT error = " << right_reproj_error << "\n\n";
      cv::waitKey(0);
    //}
    }
    if (num >= 0)
      break;
  }
}

void EvalHelper::DrawDeformationFieldParams(const int bin_rows, const int bin_cols,
                                            const double* left_dx, const double* left_dy,
                                            const double* right_dx, const double* right_dy,
                                            const uint64_t* left_num_points, const uint64_t* right_num_points, 
                                            cv::Mat& img)
{
  double bin_height = (double)img.rows / bin_rows;
  double bin_width = (double)img.cols / bin_cols;

  double font_size = 0.7;
  int thickness = 1;
  //cv::Scalar color_text(255,0,0);
  cv::Scalar color_text(0,255,0);
  // scale = 1.2
  //cv::Point text_shift1(-15,20);
  //cv::Point text_shift2(0,20);
  // scale = 1.8
  cv::Point text_shift1(-10,20);
  cv::Point text_shift2(0,15);

  for(int i = 0; i < bin_rows; i++) {
    for(int j = 0; j < bin_cols; j++) {
      int bin_num = i*bin_cols + j;
      //double red, green, blue;
      //cv::Scalar rect_color(blue, green, red);
      cv::Scalar rect_color(0,0,0);
      cv::Point pt1, pt2;
      pt1.x = j*bin_width;
      pt1.y = i*bin_height;
      pt2.x = pt1.x + bin_width;
      pt2.y = pt1.y + bin_height;
      cv::rectangle(img, pt1, pt2, rect_color, CV_FILLED);
      
      ////assert(reproj_errors_left[bin_num].size() == reproj_errors_right[bin_num].size());
      int left_n = left_num_points[bin_num];
      int right_n = right_num_points[bin_num];
      cv::Point text_pos;
      //double text_start = 4.0; // for 1.2 scale
      double text_start = 6.0; // for 1.8 scale
      //text_pos.x = j*bin_width + bin_width/6.0;
      text_pos.x = j*bin_width + bin_width/14.0;
      text_pos.y = i*bin_height + bin_height/text_start;
      //int font_type = cv::FONT_HERSHEY_SIMPLEX;
      int font_type = cv::FONT_HERSHEY_PLAIN;
      cv::putText(img, "L_N= " + std::to_string(left_n), text_pos,
                  font_type, font_size, color_text, thickness);
      text_pos += text_shift2;
      cv::putText(img, "R_N= " + std::to_string(right_n), text_pos,
                  font_type, font_size, color_text, thickness);
      text_pos += text_shift2;
      cv::putText(img, "ldx= " + std::to_string(left_dx[bin_num]), text_pos,
                  font_type, font_size, color_text, thickness);
      text_pos += text_shift2;
      cv::putText(img, "ldy= " + std::to_string(left_dy[bin_num]), text_pos,
                  font_type, font_size, color_text, thickness);
      text_pos += text_shift2;
      cv::putText(img, "rdx= " + std::to_string(right_dx[bin_num]), text_pos,
                  font_type, font_size, color_text, thickness);
      text_pos += text_shift2;
      cv::putText(img, "rdy= " + std::to_string(right_dy[bin_num]), text_pos,
                  font_type, font_size, color_text, thickness);
    }
  }
}


void EvalHelper::CalculateReprojectionErrors(track::StereoTrackerBase& tracker, const double* cam_params,
                                             const cv::Mat& cvRt, const int img_rows, const int img_cols,
                                             std::vector<std::vector<double>>& reproj_errors_left,
                                             std::vector<std::vector<double>>& reproj_errors_right,
                                             std::vector<std::vector<Eigen::Vector2d>>& left_error_vectors,
                                             std::vector<std::vector<Eigen::Vector2d>>& right_error_vectors,
                                             int h_bins, int v_bins)
{
  double bin_width = (double)img_cols / h_bins;
  double bin_height = (double)img_rows / v_bins;
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);
  double left_reproj_error, right_reproj_error;

  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if(pt_left.age_ < 1) continue;

    Eigen::Vector2d left_vec_error, right_vec_error;
    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          left_reproj_error, right_reproj_error, left_vec_error, right_vec_error);

    // add errors form 3 points in left camera and form 3 points in right camera

    left_reproj_error /= 3.0;
    right_reproj_error /= 3.0;
    left_vec_error /= 3.0;
    right_vec_error /= 3.0;
    // left prev is responsible for both errors
    int c = pt_left.prev_.x_ / bin_width;
    int r = pt_left.prev_.y_ / bin_height;
    reproj_errors_left[r*h_bins + c].push_back(left_reproj_error);
    left_error_vectors[r*h_bins + c].push_back(left_vec_error);
    reproj_errors_left[r*h_bins + c].push_back(right_reproj_error);
    left_error_vectors[r*h_bins + c].push_back(right_vec_error);

    // right prev is responsible for both errors
    c = pt_right.prev_.x_ / bin_width;
    r = pt_right.prev_.y_ / bin_height;
    reproj_errors_right[r*h_bins + c].push_back(right_reproj_error);
    right_error_vectors[r*h_bins + c].push_back(right_vec_error);
    reproj_errors_right[r*h_bins + c].push_back(left_reproj_error);
    right_error_vectors[r*h_bins + c].push_back(left_vec_error);

    // left curr responsible only for left error
    c = pt_left.curr_.x_ / bin_width;
    r = pt_left.curr_.y_ / bin_height;
    reproj_errors_left[r*h_bins + c].push_back(left_reproj_error);
    left_error_vectors[r*h_bins + c].push_back(left_vec_error);

    // right curr responsible only for right error
    c = pt_right.curr_.x_ / bin_width;
    r = pt_right.curr_.y_ / bin_height;
    reproj_errors_right[r*h_bins + c].push_back(right_reproj_error);
    right_error_vectors[r*h_bins + c].push_back(right_vec_error);
  }
}

void EvalHelper::GetStereoReprojErrors(const core::Point& pt_left_prev, const core::Point& pt_right_prev,
                                       const core::Point& pt_left_curr, const core::Point& pt_right_curr,
                                       const Eigen::Matrix4d& Rt, const double* cam_params,
                                       double& left_error, double& right_error,
                                       Eigen::Vector2d& left_vec_error, Eigen::Vector2d& right_vec_error)
{
  //std::cout << "LP = " << pt_lp << "\nRP = " << pt_rp << "\nLC = " << pt_lc << "\nRC = " << pt_rc << '\n';
  Eigen::Vector4d lp_track, lc_track;
  core::Point proj_left, proj_right;
  core::MathHelper::triangulate(cam_params, pt_left_prev, pt_right_prev, lp_track);
  //core::MathHelper::triangulate(cam_params, pt_left_curr, pt_right_curr, lc_track);
  Eigen::Vector4d curr_gt = Rt * lp_track;
  core::MathHelper::projectToStereo(cam_params, curr_gt, proj_left, proj_right);
  //error_3d = (curr_gt - lc_track).norm();
  left_error = core::MathHelper::getDist2D(proj_left, pt_left_curr);
  right_error = core::MathHelper::getDist2D(proj_right, pt_right_curr);
  left_vec_error[0] = proj_left.x_ - pt_left_curr.x_;
  left_vec_error[1] = proj_left.y_ - pt_left_curr.y_;
  right_vec_error[0] = proj_right.x_ - pt_right_curr.x_;
  right_vec_error[1] = proj_right.y_ - pt_right_curr.y_;
}

void EvalHelper::GetStereoReprojErrors(const core::Point& pt_left_prev, const core::Point& pt_right_prev,
                                       const core::Point& pt_left_curr, const core::Point& pt_right_curr,
                                       const Eigen::Matrix4d& Rt, const double* cam_params,
                                       double& error_3d, double& left_error, double& right_error)
{
  //std::cout << "LP = " << pt_lp << "\nRP = " << pt_rp << "\nLC = " << pt_lc << "\nRC = " << pt_rc << '\n';
  Eigen::Vector4d lp_track, lc_track;
  core::Point proj_left, proj_right;
  core::MathHelper::triangulate(cam_params, pt_left_prev, pt_right_prev, lp_track);
  core::MathHelper::triangulate(cam_params, pt_left_curr, pt_right_curr, lc_track);
  Eigen::Vector4d curr_gt = Rt * lp_track;
  core::MathHelper::projectToStereo(cam_params, curr_gt, proj_left, proj_right);
  error_3d = (curr_gt - lc_track).norm();
  left_error = core::MathHelper::getDist2D(proj_left, pt_left_curr);
  right_error = core::MathHelper::getDist2D(proj_right, pt_right_curr);
}

void EvalHelper::CalculateErrorStatistics(const std::vector<std::vector<double>>& reproj_errors,
                                          const std::vector<std::vector<Eigen::Vector2d>>& error_vectors,
                                          std::vector<double>& means, std::vector<double>& variances,
                                          std::vector<Eigen::Vector2d>& vec_means,
                                          std::vector<Eigen::Vector2d>& vec_variances)
{
  int num_bins = reproj_errors.size();
  means.assign(num_bins, 0.0);
  variances.assign(num_bins, 0.0);
  vec_means.assign(num_bins, Eigen::Vector2d(0.0, 0.0));
  vec_variances.assign(num_bins, Eigen::Vector2d(0.0, 0.0));
  for(int i = 0; i < num_bins; i++) {
    int N = reproj_errors[i].size();
    if(N > 0) {
      for(int j = 0; j < N; j++) {
        // TODO how to choose the left and right bins
        means[i] += reproj_errors[i][j];
        vec_means[i] += error_vectors[i][j];
      }
      means[i] /= (double) N;
      vec_means[i] = vec_means[i] / (double) N;
      if(N > 0) { 
        for(int j = 0; j < N; j++) {
          double norm_err = (reproj_errors[i][j] - means[i]);
          variances[i] += norm_err*norm_err;
          norm_err = (reproj_errors[i][j] - means[i]);
          variances[i] += norm_err*norm_err;

          Eigen::Vector2d vec_err = error_vectors[i][j] - vec_means[i];
          vec_variances[i] += vec_err.cwiseProduct(vec_err);
          vec_err = error_vectors[i][j] - vec_means[i];
          vec_variances[i] += vec_err.cwiseProduct(vec_err);
        }
        variances[i] /= static_cast<double>(N - 1);
        vec_variances[i] /= static_cast<double>(N - 1);
      }
    }
    // it is empty
    else
      means[i] = -std::numeric_limits<double>::max();
    //printf("bin: %d, N = %d, mean = %f\n", i, N, means[i]);
  }
  //for(size_t i = 0;  i < left_reproj_errors.size(); i++) {
  //  std::cout << "Bin: " << i << "\n\tmean = " << means[i] << "\n\tvariance = " << variances[i] << "\n";
  //  std::cout << "\tu_mean = " << vec_means[i][0] << "\n\tv_mean = " << vec_means[i][1] << "\n";
  //}
}

void EvalHelper::CalculateErrorStatistics(const std::vector<std::vector<double>>& left_reproj_errors,
                                          const std::vector<std::vector<double>>& right_reproj_errors,
                                          const std::vector<std::vector<Eigen::Vector2d>>& left_error_vectors,
                                          const std::vector<std::vector<Eigen::Vector2d>>& right_error_vectors,
                                          std::vector<double>& means, std::vector<double>& variances,
                                          std::vector<Eigen::Vector2d>& vec_means,
                                          std::vector<Eigen::Vector2d>& vec_variances)
{
  int num_bins = left_reproj_errors.size();
  means.assign(num_bins, 0.0);
  variances.assign(num_bins, 0.0);
  vec_means.assign(num_bins, Eigen::Vector2d(0.0, 0.0));
  vec_variances.assign(num_bins, Eigen::Vector2d(0.0, 0.0));
  for(int i = 0; i < num_bins; i++) {
    int N = left_reproj_errors[i].size();
    if(N > 0) {
      for(int j = 0; j < N; j++) {
        // TODO how to choose the left and right bins
        means[i] += left_reproj_errors[i][j];
        means[i] += right_reproj_errors[i][j];
        vec_means[i] += left_error_vectors[i][j];
        vec_means[i] += right_error_vectors[i][j];
      }
      means[i] /= 2.0 * N;
      vec_means[i] = vec_means[i] / (2.0 * N);
      if(N > 0) { 
        for(int j = 0; j < N; j++) {
          double norm_err = (left_reproj_errors[i][j] - means[i]);
          variances[i] += norm_err*norm_err;
          norm_err = (right_reproj_errors[i][j] - means[i]);
          variances[i] += norm_err*norm_err;

          Eigen::Vector2d vec_err = left_error_vectors[i][j] - vec_means[i];
          vec_variances[i] += vec_err.cwiseProduct(vec_err);
          vec_err = right_error_vectors[i][j] - vec_means[i];
          vec_variances[i] += vec_err.cwiseProduct(vec_err);
        }
        variances[i] /= static_cast<double>(2*N - 1);
        vec_variances[i] /= static_cast<double>(2*N - 1);
      }
    }
    // it is empty
    else
      means[i] = -std::numeric_limits<double>::max();
    //printf("bin: %d, N = %d, mean = %f\n", i, N, means[i]);
  }
  //for(size_t i = 0;  i < left_reproj_errors.size(); i++) {
  //  std::cout << "Bin: " << i << "\n\tmean = " << means[i] << "\n\tvariance = " << variances[i] << "\n";
  //  std::cout << "\tu_mean = " << vec_means[i][0] << "\n\tv_mean = " << vec_means[i][1] << "\n";
  //}
}

void EvalHelper::SaveErrorStatistics(const int rows, const int cols,
                                     const std::vector<std::vector<double>>& reproj_errors,
                                     const std::vector<double>& means, const std::vector<double>& variances,
                                     const std::vector<Eigen::Vector2d>& vec_means,
                                     const std::vector<Eigen::Vector2d>& vec_variances,
                                     const std::string file_sufix)
{
  cv::Mat mean_mat(rows, cols, CV_64F);
  cv::Mat variance_mat(rows, cols, CV_64F);
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      int bin_num = i*cols + j;
      mean_mat.at<double>(i,j) = means[bin_num];
      variance_mat.at<double>(i,j) = variances[bin_num];
    }
  }
  cv::FileStorage mat_file("mean_" + file_sufix, cv::FileStorage::WRITE);
  mat_file << "mean_matrix" << mean_mat;
  mat_file.open("variance_" + file_sufix, cv::FileStorage::WRITE);
  mat_file << "variance_matrix" << variance_mat;

  std::ofstream n_file("statistics_num.txt");
  std::ofstream mean_file("statistics_mean.txt");
  std::ofstream mean_x_file("statistics_mean_x.txt");
  std::ofstream mean_y_file("statistics_mean_y.txt");
  std::ofstream variance_file("statistics_variance.txt");
  std::ofstream variance_x_file("statistics_variance_x.txt");
  std::ofstream variance_y_file("statistics_variance_y.txt");
  //std::ofstream mean_hist_file("mean_histogram.txt");
  //double mean_sum = 0.0;
  //for(int j = 0; j < h_bins; j++) {
  //  for(int i = 0; i < v_bins; i++) {
  //    int bin_num = i*h_bins + j;
  //    double mean = means[bin_num];
  //    mean_sum += mean;
  //  }
  //  mean_hist_file << mean_sum << '\n';
  //  mean_sum = 0.0;
  //}
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      int bin_num = i*cols + j;
      int N = reproj_errors[bin_num].size();
      double mean = means[bin_num];
      double mean_x = vec_means[bin_num][0];
      double mean_y = vec_means[bin_num][1];
      double variance = variances[bin_num];
      double var_x = vec_variances[bin_num][0];
      double var_y = vec_variances[bin_num][1];
      n_file << N;
      mean_file << std::right << std::setprecision(3) << mean;
      mean_x_file << std::right << std::setprecision(3) << mean_x;
      mean_y_file << std::right << std::setprecision(3) << mean_y;
      variance_file << std::right << std::setprecision(3) << variance;
      variance_x_file << std::right << std::setprecision(3) << var_x;
      variance_y_file << std::right << std::setprecision(3) << var_y;
      if(j < (cols-1)) {
        n_file << " & ";
        mean_file << " & ";
        mean_x_file << " & ";
        mean_y_file << " & ";
        variance_file << " & ";
        variance_x_file << " & ";
        variance_y_file << " & ";
      }
    }
    n_file << " \\\\\n";
    mean_file << " \\\\\n";
    mean_x_file << " \\\\\n";
    mean_y_file << " \\\\\n";
    variance_file << " \\\\\n";
    variance_x_file << " \\\\\n";
    variance_y_file << " \\\\\n";
  }
}

//void EvalHelper::SaveErrorStatistics(const int rows, const int cols,
//                                     const std::vector<double>& means, const std::vector<double>& variances,
//                                     const std::vector<Eigen::Vector2d>& vec_means,
//                                     const std::vector<Eigen::Vector2d>& vec_variances,
//                                     const std::string file_name)
//{
//  cv::Mat stats_mat(rows, cols, CV_64FC(6));
//  for(int i = 0; i < rows; i++) {
//    for(int j = 0; j < cols; j++) {
//      int bin_num = i*cols + j;
//      stats_mat.at<Vec6d>(i,j)[0] = means[bin_num];
//      stats_mat.at<Vec6d>(i,j)[1] = variances[bin_num];
//      stats_mat.at<Vec6d>(i,j)[2] = vec_means[bin_num][0];
//      stats_mat.at<Vec6d>(i,j)[3] = vec_means[bin_num][1];
//      stats_mat.at<Vec6d>(i,j)[4] = vec_variances[bin_num][0];
//      stats_mat.at<Vec6d>(i,j)[5] = vec_variances[bin_num][1];
//    }
//  }
//  cv::FileStorage mat_file(file_name, cv::FileStorage::WRITE);
//  mat_file << "statistics_matrix" << stats_mat;
//}

void EvalHelper::DrawErrorStatistics(const int bin_rows, const int bin_cols,
                                     const std::vector<double>& means, const std::vector<double>& variances,
                                     const std::vector<Eigen::Vector2d>& vec_means,
                                     const std::vector<Eigen::Vector2d>& vec_variances,
                                     const std::vector<std::vector<double>>& reproj_errors,
                                     const bool draw_stats, cv::Mat& img)
{
  double bin_height = (double)img.rows / bin_rows;
  double bin_width = (double)img.cols / bin_cols;

  double font_size = 1.0;
  int thickness = 1;
  //cv::Scalar color_text(255,0,0);
  cv::Scalar color_text(0,255,0);
  // scale = 1.2
  //cv::Point text_shift1(-15,20);
  //cv::Point text_shift2(0,20);
  // scale = 1.8
  cv::Point text_shift1(-10,20);
  cv::Point text_shift2(0,15);


  double mmin = std::numeric_limits<double>::max();
  double mmax = -std::numeric_limits<double>::max();
  int num_bins = means.size();
  for(int i = 0; i < num_bins; i++) {
    if(means[i] >= 0.0) {
      if(means[i] > mmax)
        mmax = means[i];
      if(means[i] < mmin)
        mmin = means[i];
    }
  }
  double mdiff = mmax - mmin;
  //std::cout << "mdiff = " << mdiff << '\n';

  for(int i = 0; i < bin_rows; i++) {
    for(int j = 0; j < bin_cols; j++) {
      double red, green, blue;
      int bin_num = i*bin_cols + j;
      double mean = means[bin_num];
      if(mean > 5.0) {
        std::cout << "\033[1;31m BIG MEAN = " << mean << "\033[0m\n";
        cv::waitKey(0);
        throw 1;
      }
      if(mean >= 0.0) {
        double gray = ((mean - mmin) / mdiff) * 255.0;
        red = gray;
        green = gray;
        blue = gray;
      }
      // if empty
      else {
        red = 0.0;
        green = 0.0;
        blue = 255.0;
      }
      //double white = ((mmax - mean) / mdiff) * 255.0;
      // supress green color
      //double green = 0.7 * ((mmax - mean) / mdiff) * 255.0;

      cv::Scalar rect_color(blue, green, red);
      cv::Point pt1, pt2;
      pt1.x = j*bin_width;
      pt1.y = i*bin_height;
      pt2.x = pt1.x + bin_width;
      pt2.y = pt1.y + bin_height;
      cv::rectangle(img, pt1, pt2, rect_color, CV_FILLED);
      
      if(draw_stats) {
        int N = reproj_errors[bin_num].size();
        cv::Point text_pos;
        //double text_start = 4.0; // for 1.2 scale
        double text_start = 6.0; // for 1.8 scale
        text_pos.x = j*bin_width + bin_width/6.0;
        text_pos.y = i*bin_height + bin_height/text_start;
        double variance = variances[bin_num];
        //int font_type = cv::FONT_HERSHEY_SIMPLEX;
        int font_type = cv::FONT_HERSHEY_PLAIN;
        cv::putText(img, "N= " + std::to_string(N), text_pos,
                    font_type, font_size, color_text, thickness);
        text_pos += text_shift1;
        cv::putText(img, "m= " + std::to_string(mean), text_pos,
                    font_type, font_size, color_text, thickness);
        text_pos += text_shift2;
        cv::putText(img, "v= " + std::to_string(variance), text_pos,
                    font_type, font_size, color_text, thickness);
        text_pos += text_shift2;
        cv::putText(img, "xm= " + std::to_string(vec_means[bin_num][0]), text_pos,
                    font_type, font_size, color_text, thickness);
        text_pos += text_shift2;
        cv::putText(img, "ym= " + std::to_string(vec_means[bin_num][1]), text_pos,
                    font_type, font_size, color_text, thickness);
        text_pos += text_shift2;
        cv::putText(img, "xv= " + std::to_string(vec_variances[bin_num][0]), text_pos,
                    font_type, font_size, color_text, thickness);
        text_pos += text_shift2;
        cv::putText(img, "yv= " + std::to_string(vec_variances[bin_num][1]), text_pos,
                    font_type, font_size, color_text, thickness);
      }
    }
  }
}

int EvalHelper::CountFilterStoreBadTracks(track::StereoTrackerBase& tracker, const double* cam_params,
      const cv::Mat& cvRt, bool save_patches, const std::string& good_folder, const std::string& bad_folder,
      std::vector<size_t>& track_index, std::vector<size_t>& track_cnt, size_t& all_tracks_cnt,
      bool filter_bad, double reproj_error_thr, double max_remove_ratio)
{
  Eigen::Vector4d lp_track, lc_track;
  Eigen::Matrix4d Rt;
  core::Point proj_left, proj_right;
  cv::cv2eigen(cvRt, Rt);
  //std::cout << "GT motion:\n" << Rt << "\n";
  
  double error_3d, left_reproj_error, right_reproj_error;
  std::vector<size_t> bad_tracks;
  std::vector<double> bad_tracks_errors;

  std::string save_folder;
  bool is_bad;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if(pt_left.age_ < 1) continue;
    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          error_3d, left_reproj_error, right_reproj_error);

    // use depth error in prev frame also
    // cant because of to much false positives
    //double f = cam_params[0];
    //double baseline = cam_params[4];
    //double lx = pt_left.prev_.x_;
    //double ly = pt_left.prev_.y_;
    //double depth = triangulateDepth(pt_left.prev_, pt_right.prev_, f, baseline);
    //int row = (int)std::round(ly);
    //int col = (int)std::round(lx);
    //double depth_error = std::abs(depth - depth_prev.at<float>(row, col));
    //if(depth_error < 100.0) continue;

    // TODO which are bad
    //if(error < 10.0 && left_reproj_error < 1.0 && right_reproj_error < 1.0) continue;
    if(left_reproj_error < reproj_error_thr && right_reproj_error < reproj_error_thr) {
      save_folder = good_folder;
      is_bad = false;
    }
    else {
      save_folder = bad_folder;
      is_bad = true;
    }

    // only for plain patch type tracker
    if(save_patches) {
      track::FeatureData left_data = tracker.getLeftFeatureData(i);
      track::FeatureData right_data = tracker.getRightFeatureData(i);
      if(pt_left.age_ == 1) {
        track_index[i] = all_tracks_cnt++;
        track_cnt[i] = 0;
        std::stringstream prefix;
        prefix << setfill('0') << setw(6) << std::to_string(track_index[i]) << "_"
               << setfill('0') << setw(3) << std::to_string(track_cnt[i]);
        track_cnt[i]++;
        save_patch(save_folder, left_data.desc_prev_, prefix.str() + "_left.pgm");
        save_patch(save_folder, right_data.desc_prev_, prefix.str() + "_right.pgm");
      }
      std::stringstream prefix;
      prefix << setfill('0') << setw(6) << std::to_string(track_index[i]) << "_"
             << setfill('0') << setw(3) << std::to_string(track_cnt[i]);
      track_cnt[i]++;
      save_patch(save_folder, left_data.desc_curr_, prefix.str() + "_left.pgm");
      save_patch(save_folder, right_data.desc_curr_, prefix.str() + "_right.pgm");
    }

    if(filter_bad) {
      if(!is_bad) continue;
      bad_tracks.push_back(i);
      bad_tracks_errors.push_back(std::max(left_reproj_error, right_reproj_error));
    }

    //std::cout << "Left prev:\n" << lp_track << "\nLeft curr:\n" << lc_track << "\n";
    //std::cout << "Pt left curr GT:\n" << curr_gt << "\n";
    //std::cout << "Prev depth error = " << depth_error << "\n";

    //std::cout << "Track: " << i << "\n";
    //std::cout << "Img left:\n" << pt_left.curr_ << "\n";
    //std::cout << "Proj left:\n" << proj_left << "\n";
    //std::cout << "Img right:\n" << pt_right.curr_ << "\n";
    //std::cout << "Proj right:\n" << proj_right << "\n";
    //std::cout << "Curr 3D euclid error = " << error << "\n";
    //std::cout << "Curr 2D reproj error left = " << left_reproj_error << "\n";
    //std::cout << "Curr 2D reproj error right = " << right_reproj_error << "\n-----\n";
    //tracker.showTrack(i);
  }

  if(filter_bad) {
    std::vector<size_t> indices(bad_tracks.size());
    std::iota(std::begin(indices), std::end(indices), static_cast<size_t>(0));
    std::sort(indices.begin(), indices.end(),
        [&](size_t a, size_t b) { return bad_tracks_errors[a] > bad_tracks_errors[b]; } );

    //std::unordered_set<size_t> removed;
    // remove only remove_percent % of all tracks
    size_t remove_limit = max_remove_ratio * tracker.countActiveTracks();
    size_t i;
    for(i = 0; i < bad_tracks.size(); i++) {
      size_t bad_idx = indices[i];
      tracker.removeTrack(bad_tracks[bad_idx]);
      //printf("Deleted track with Reproj error = %f\n", bad_tracks_errors[bad_idx]);
      if(i >= remove_limit) {
        i++;
        break;
      }
      //if(removed.find(bad_tracks[i]) == removed.end()) {
      //  printf("Deleted track with Reproj error = %f\n", bad_tracks_errors[bad_tracks[i]]);
      //  tracker.removeTrack(bad_tracks[i]);
      //  removed.insert(bad_tracks[i]);
      //}
      //if(removed.size() > remove_limit)
      //  break;
    }
    printf("Num of deleted bad tracks = %ld\n", i);
  }
  return bad_tracks.size();
}


void EvalHelper::DrawTracksAndErrors(track::StereoTrackerBase& tracker, const double* cam_params,
                                     const cv::Mat& cvRt, const cv::Mat& img_lp, const cv::Mat& img_rp,
                                     const std::string& left_name,
                                     bool save_errors,
                                     std::vector<std::vector<double>>& reproj_errors_left,
                                     std::vector<std::vector<double>>& reproj_errors_right,
                                     int h_bins, int v_bins, bool draw_on, double thr, bool filter_bad)
{
  double bin_width = (double)img_lp.cols / h_bins;
  double bin_height = (double)img_lp.rows / v_bins;
  Eigen::Matrix4d Rt;
  cv::cv2eigen(cvRt, Rt);
  double error_3d, left_reproj_error, right_reproj_error;

  int num_feats = 0;
  //double scale = 8.0;
  double scale = 1.0;
  double font_size = 0.3;
  int thickness = 1;
  cv::Point shift_pt(scale/2, scale/2);
  cv::Point shift_down(0, scale);
  cv::Scalar color_prev(0,0,255);
  cv::Scalar color_text(0,0,255);
  cv::Mat disp_lp, disp_rp;
  if(draw_on) {
    cv::cvtColor(img_lp, disp_lp, cv::COLOR_GRAY2RGB);
    cv::cvtColor(img_rp, disp_rp, cv::COLOR_GRAY2RGB);
    cv::Point central_point;
    central_point.x = std::round(cam_params[2]);
    central_point.y = std::round(cam_params[3]);
    cv::circle(disp_lp, central_point, 4, cv::Scalar(0,255,0), -1);
    cv::resize(disp_lp, disp_lp, cv::Size(), scale, scale);
  }
  //bool has_big_errors = false;

  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo pt_left = tracker.featureLeft(i);
    track::FeatureInfo pt_right = tracker.featureRight(i);
    if(pt_left.age_ < 1) continue;

    GetStereoReprojErrors(pt_left.prev_, pt_right.prev_, pt_left.curr_, pt_right.curr_, Rt, cam_params,
                          error_3d, left_reproj_error, right_reproj_error);
    if(save_errors) {
      int c = pt_left.prev_.x_ / bin_width;
      int r = pt_left.prev_.y_ / bin_height;
      reproj_errors_left[r*h_bins + c].push_back(left_reproj_error);
      c = pt_right.prev_.x_ / bin_width;
      r = pt_right.prev_.y_ / bin_height;
      reproj_errors_right[r*h_bins + c].push_back(right_reproj_error);
    }

    //double thr = 0.0;
    //if(left_reproj_error > thr || right_reproj_error > thr) {
    //double thr = 4.0;
    //if(draw_on) {
    if(left_reproj_error < thr && right_reproj_error < thr) {
      num_feats++;
      //has_big_errors = true;
      if(draw_on) {
        cv::Point pt_lp;
        pt_lp.x = (int)std::round(scale * pt_left.prev_.x_);
        pt_lp.y = (int)std::round(scale * pt_left.prev_.y_);
        //cv::line(img_rc, pt1, pt2, color_prev, 2, 8);
        cv::circle(disp_lp, pt_lp, 2, color_prev, -1);
        cv::putText(disp_lp, std::to_string(left_reproj_error).substr(0,4), pt_lp + shift_pt,
                    cv::FONT_HERSHEY_SIMPLEX, font_size, color_text, thickness);
        cv::putText(disp_lp, std::to_string(right_reproj_error).substr(0,4), pt_lp + shift_pt + shift_down,
                    cv::FONT_HERSHEY_SIMPLEX, font_size, color_text, thickness);
        cv::putText(disp_lp, std::to_string(i), pt_lp + shift_pt + 2*shift_down, cv::FONT_HERSHEY_SIMPLEX,
                    font_size, color_text, thickness);
      }
    }
    else if(filter_bad)
      tracker.removeTrack(i);
  }
  std::cout << "[EvalHelper]: Num of feats below error (" << thr << ") = " << num_feats << '\n';

  if(draw_on) {
  //if(draw_on && has_big_errors) {
    cv::imshow(left_name, disp_lp);
    //while(cv::waitKey(0) != 27);
  }
}

void EvalHelper::DrawErrorDistribution(int h_bins, int v_bins,
                                       std::vector<std::vector<double>>& reproj_errors,
                                       cv::Mat& img, bool draw_stats, bool save_stats)
{
  double bin_width = (double)img.cols / h_bins;
  double bin_height = (double)img.rows / v_bins;

  double font_size = 0.5;
  int thickness = 1;
  cv::Scalar color_text(0,0,255);

  double mmin = std::numeric_limits<double>::max();
  double mmax = -std::numeric_limits<double>::max();
  int num_bins = reproj_errors.size();
  std::vector<double> means;
  std::vector<double> variances;
  means.assign(num_bins, 0.0);
  variances.assign(num_bins, 0.0);
  for(int i = 0; i < num_bins; i++) {
    int N = reproj_errors[i].size();
    if(N > 0) {
      for(int j = 0; j < N; j++)
        means[i] += reproj_errors[i][j];
      means[i] /= (double)N;
      for(int j = 0; j < N; j++) {
        double norm_err = (reproj_errors[i][j] - means[i]);
        variances[i] += norm_err*norm_err;
      }
      if(N > 1)
        variances[i] /= (double)(N-1);

      if(means[i] > mmax)
        mmax = means[i];
      if(means[i] < mmin)
        mmin = means[i];
    }
    // it is empty
    else
      means[i] = -std::numeric_limits<double>::max();
    //printf("bin: %d, N = %d, mean = %f\n", i, N, means[i]);
  }
  double mdiff = mmax - mmin;
  //std::cout << "mdiff = " << mdiff << '\n';

  for(int i = 0; i < v_bins; i++) {
    for(int j = 0; j < h_bins; j++) {
      double red, green, blue;
      int bin_num = i*h_bins + j;
      double mean = means[bin_num];
      if(mean > 5.0) {
        std::cout << "\033[1;31m BIG MEAN = " << mean << "\033[0m\n";
        cv::waitKey(0);
        throw 1;
      }
      if(mean >= 0.0) {
        double gray = ((mean - mmin) / mdiff) * 255.0;
        red = gray;
        green = gray;
        blue = gray;
      }
      // if empty
      else {
        red = 0.0;
        green = 0.0;
        blue = 255.0;
      }
      //double white = ((mmax - mean) / mdiff) * 255.0;
      // supress green color
      //double green = 0.7 * ((mmax - mean) / mdiff) * 255.0;

      cv::Scalar rect_color(blue, green, red);
      cv::Point pt1, pt2;
      pt1.x = j*bin_width;
      pt1.y = i*bin_height;
      pt2.x = pt1.x + bin_width;
      pt2.y = pt1.y + bin_height;
      cv::rectangle(img, pt1, pt2, rect_color, CV_FILLED);
      
      if(draw_stats) {
        int N = reproj_errors[bin_num].size();
        cv::Point text_pos;
        text_pos.x = j*bin_width + bin_width/4.0;
        //text_pos.y = i*bin_height + bin_height/1.9;
        text_pos.y = i*bin_height + bin_height/4.0;
        cv::Point text_shift1(-15,20);
        cv::Point text_shift2(0,20);
        double variance = variances[bin_num];
        cv::putText(img, std::to_string(N), text_pos, cv::FONT_HERSHEY_SIMPLEX,
                    font_size, color_text, thickness);
        text_pos += text_shift1;
        cv::putText(img, std::to_string(mean), text_pos, cv::FONT_HERSHEY_SIMPLEX,
                    font_size, color_text, thickness);
        text_pos += text_shift2;
        cv::putText(img, std::to_string(variance), text_pos, cv::FONT_HERSHEY_SIMPLEX,
                    font_size, color_text, thickness);
      }
    }
  }

  if(save_stats) {
    std::ofstream mean_hist_file("mean_histogram.txt");
    std::ofstream n_file("statistics_num.txt");
    std::ofstream mean_file("statistics_mean.txt");
    std::ofstream variance_file("statistics_variance.txt");
    double mean_sum = 0.0;
    for(int j = 0; j < h_bins; j++) {
      for(int i = 0; i < v_bins; i++) {
        int bin_num = i*h_bins + j;
        double mean = means[bin_num];
        mean_sum += mean;
      }
      mean_hist_file << mean_sum << '\n';
      mean_sum = 0.0;
    }
    for(int i = 0; i < v_bins; i++) {
      for(int j = 0; j < h_bins; j++) {
        int bin_num = i*h_bins + j;
        int N = reproj_errors[bin_num].size();
        double mean = means[bin_num];
        double variance = variances[bin_num];
        n_file << N;
        mean_file << std::right << std::setprecision(3) << mean;
        variance_file << std::right << std::setprecision(3) << variance;
        if(j < (h_bins-1)) {
          n_file << " & ";
          mean_file << " & ";
          variance_file << " & ";
        }
      }
      n_file << " \\\\\n";
      mean_file << " \\\\\n";
      variance_file << " \\\\\n";
    }
  }

  cv::imshow("error_distribution", img);
  cv::waitKey(100);
  //cv::waitKey(0);
}

void EvalHelper::voPoint2cvMat(core::Point& pt, cv::Mat& mat)
{
  point_to_cvmat(pt, mat);
}


void EvalHelper::DrawStereoTracks(StereoTrackerBase& tracker,
                                  const cv::Mat& img_left,
                                  const cv::Mat& img_right,
                                  const std::string& name_left,
                                  const std::string& name_right)
{
  cv::Mat disp_left, disp_right;
  cv::cvtColor(img_left, disp_left, cv::COLOR_GRAY2RGB);
  cv::cvtColor(img_right, disp_right, cv::COLOR_GRAY2RGB);
  double font_size = 0.4; // 0.3
  cv::Point pt1, pt2;
  cv::Point pt_shift(0,5);
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);
  //double scale = 5.0;
  double scale = 1.0;
  int line_thickness = 1;
  cv::resize(disp_left, disp_left, cv::Size(), scale, scale);  
  static double max_dist_x = 0.0;
  static double max_dist_y = 0.0;
  //double max_dist_prev_x = max_dist_x;
  //double max_dist_prev_y = max_dist_y;
  //bool draw_on = false;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    // 3  56  83  120  164  495  573  575  Avg. age: 2.58657
    //if(i >= 0 || i == 120 || i == 573 || i == 575) {

    FeatureInfo feat_left = tracker.featureLeft(i);
    FeatureInfo feat_right = tracker.featureRight(i);

    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      //double disp1 = feat_left.prev_.x_ - feat_right.prev_.x_;
      //double disp2 = feat_left.curr_.x_ - feat_right.curr_.x_;
      //double disp_diff = (disp2 - disp1);
      //std::cout << "disp_diff = " << disp_diff << "\n";
      //if(disp_diff <= 5 && disp_diff >= 0.0) continue;

      //max_dist_prev_x = max_dist_x;
      //max_dist_prev_y = max_dist_y;
      max_dist_x = std::max(max_dist_x, std::abs(feat_left.prev_.x_ - feat_left.curr_.x_));
      max_dist_y = std::max(max_dist_y, std::abs(feat_left.prev_.y_ - feat_left.curr_.y_));
      //if(!(max_dist_x > max_dist_prev_x || max_dist_y > max_dist_prev_y)) {
      //  continue;
      //}
      //else
      //  draw_on = true;
      pt1.x = scale * feat_left.prev_.x_;
      pt1.y = scale * feat_left.prev_.y_;
      pt2.x = scale * feat_left.curr_.x_;
      pt2.y = scale * feat_left.curr_.y_;
      cv::line(disp_left, pt1, pt2, color_prev, line_thickness);
      cv::circle(disp_left, pt1, 2, color_prev, -1);
      cv::circle(disp_left, pt2, 2, color_curr, -1);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      cv::putText(disp_left, to_string(i), pt1 + pt_shift, cv::FONT_HERSHEY_SIMPLEX, font_size,
                  cv::Scalar(0,0,255), 1, 8);

      pt1.x = feat_right.prev_.x_;
      pt1.y = feat_right.prev_.y_;
      pt2.x = feat_right.curr_.x_;
      pt2.y = feat_right.curr_.y_;
      cv::line(disp_right, pt1, pt2, color_prev, line_thickness);
      cv::circle(disp_right, pt1, 2, color_prev, -1);
      cv::circle(disp_right, pt2, 2, color_curr, -1);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      //cv::putText(img_rc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);
      //cv::imshow("left_prev_track", img_lc);
      //cv::imshow("right_prev_track", img_rc);
      //cv::waitKey(0);
    }

    //}
  }
  std::cout << "MAX dist x = " << max_dist_x << "\nMAX dist y = " << max_dist_y << "\n\n";
  cv::imshow(name_left, disp_left);
  cv::imshow(name_right, disp_right);
  //if(max_dist_x > max_dist_prev_x || max_dist_y > max_dist_prev_y)
  //if(draw_on && max_dist_x > 100)
  //  cv::waitKey(0);

  //cv::imwrite("tracks_left.png", img_lc, compression_params);
  //cv::imwrite("tracks_right.png", img_rc, compression_params);
}

double EvalHelper::GetStereoReprojError(const track::StereoTrackerBase& tracker,
                                        const double* cam_params, cv::Mat& Rt)
{
  std::vector<core::Point> points_lp, points_rp, points_lc, points_rc;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo left = tracker.featureLeft(i);
    track::FeatureInfo right = tracker.featureRight(i);

    if(left.age_ > 0) {
      points_lp.push_back(left.prev_);
      points_lc.push_back(left.curr_);
      points_rp.push_back(right.prev_);
      points_rc.push_back(right.curr_);
    }
  }

  // first triangulate image points in previous frame into 3D world points
  std::vector<cv::Mat> prev_pts3d;
  prev_pts3d.resize(points_lp.size()); // - this cant work
  double fx = cam_params[0];
  //double fy = cam_params[1];
  double cu = cam_params[2];
  double cv = cam_params[3];
  double baseline = cam_params[4];
  for(size_t i = 0; i < prev_pts3d.size(); i++) {
    double disp = points_lp[i].x_ - points_rp[i].x_;
    if(disp < 0.001) {
      std::cout << "To small disp\n";
      throw "Error";
    }
    disp = std::max(disp, 0.001);
    prev_pts3d[i] = Mat::zeros(4, 1, CV_64F);
    prev_pts3d[i].at<double>(0,0) = (points_lp[i].x_ - cu) * baseline / disp;
    prev_pts3d[i].at<double>(1,0) = (points_lp[i].y_ - cv) * baseline / disp;
    prev_pts3d[i].at<double>(2,0) = fx * baseline / disp;
    prev_pts3d[i].at<double>(3,0) = 1.0;
    //cout << prev_pts3d[i] << endl;
  }

  cv::Mat K = get_camera_matrix(cam_params);;
  //std::cout << "K = \n" << K << "\n";
  // then evaluate the reprojection error across all points
  cv::Mat curr_pts3d = Mat::zeros(4, 1, CV_64F);
  cv::Mat img_pt = Mat::zeros(3, 1, CV_64F);
  cv::Mat mat_pt2d = Mat::zeros(3, 1, CV_64F);
  double error = 0.0;
  double error_norm;
  for(size_t i = 0; i < prev_pts3d.size(); i++) {
    // transform 3d point from previous to current frame and project it to image plane
    //cout << i << ":  " << prev_pts3d[i] << endl;
    curr_pts3d = Rt * prev_pts3d[i];
    // for left camera
    img_pt = K * curr_pts3d;
    //cout << "1:\n" << img_pt << endl;
    img_pt /= img_pt.at<double>(2,0);
    //cout << img_pt << endl;

    point_to_cvmat(points_lc[i], mat_pt2d);
    error_norm = cv::norm(mat_pt2d - img_pt);
    error += error_norm * error_norm;

    // for right camera
    curr_pts3d.at<double>(0,0) -= baseline;
    img_pt = K * curr_pts3d;
    img_pt /= img_pt.at<double>(2,0);
    //cout << img_pt << endl;
    point_to_cvmat(points_rc[i], mat_pt2d);
    error_norm = cv::norm(mat_pt2d - img_pt);
    error += error_norm * error_norm;
  }
  //return error / prev_pts3d.size();
  return 0.5 * error;
}



// nice try but the depth map is not always on the right place as a feature center of filter response :/
//int EvalHelper::CountBadTracks(const track::StereoTrackerBase& tracker, const double (&cam_params)[5],
//                               const cv::Mat& cvRt, cv::Mat& depth_prev_l, cv::Mat& depth_curr_l)
//{
//  Eigen::Vector4d lp_track, lc_track;
//  Eigen::Matrix4d Rt;
//  cv::cv2eigen(cvRt, Rt);
//  for(int i = 0; i < tracker.countFeatures(); i++) {
//    track::FeatureInfo pt_left = tracker.featureLeft(i);
//    track::FeatureInfo pt_right = tracker.featureRight(i);
//    if(pt_left.age_ <= 0) continue;
//    get_3d_point(pt_left.prev_, cam_params, depth_prev_l, lp_track);
//    get_3d_point(pt_left.curr_, cam_params, depth_curr_l, lc_track);
//    Eigen::Vector4d lc_gt = Rt * lp_track;
//    std::cout << "Pt left current GT:\n" << lc_gt << "\nPt left curr tracked:\n" << lc_track << "\n";
//    double error = (lc_gt - lc_track).norm();
//    std::cout << "Left euclid error = " << error << "\n-----\n";
//    if(error > 100.0)
//      tracker.showTrack(i);
//  }
//  return 0;
//}

double EvalHelper::getStereoDepthError(track::StereoTrackerBase& tracker, const double (&cam_params)[5],
                                       const cv::Mat& depth_mat)
{
  double f = cam_params[0];
  double baseline = cam_params[4];
  double error_sum = 0.0;
  int track_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo leftf = tracker.featureLeft(i);
    track::FeatureInfo rightf = tracker.featureRight(i);
    if(leftf.age_ <= 0) continue;
    double lx = leftf.prev_.x_;
    double ly = leftf.prev_.y_;
    double depth = triangulateDepth(leftf.prev_, rightf.prev_, f, baseline);

    int row = (int)std::round(ly);
    int col = (int)std::round(lx);
    double depth_error = std::abs(depth - depth_mat.at<float>(row, col));
    //if(depth_error > DEPTH_ERR_THR) {
    //  std::cout << "[EvalHelper]: warning - skippking big depth error!\n";
    //  continue;
    //}
    error_sum += depth_error;
    track_cnt++;
  }
  return error_sum / track_cnt;
}

double EvalHelper::getStereoDepthError(track::StereoTrackerBase& tracker, const double (&cam_params)[5],
                                       const cv::Mat& depth_mat, const cv::Mat lp_img, const cv::Mat rp_img,
                                       const cv::Mat lc_img, const cv::Mat rc_img)
{
  cv::Scalar color1(255,0,0);
  double f = cam_params[0];
  double baseline = cam_params[4];
  double error_sum = 0.0;
  int track_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo leftf = tracker.featureLeft(i);
    track::FeatureInfo rightf = tracker.featureRight(i);
    if(leftf.age_ <= 0) continue;
    double lx = leftf.prev_.x_;
    double ly = leftf.prev_.y_;
    double depth = triangulateDepth(leftf.prev_, rightf.prev_, f, baseline);
    if(depth <= 0.0) continue;
    //double depth_error = std::abs(depth - depth_mat.at<float>(ly, lx));
    int row = (int)std::round(ly);
    int col = (int)std::round(lx);
    double depth_error = std::abs(depth - depth_mat.at<float>(row, col));

    if(depth_error > DRAW_ERROR_THR) {
      std::cout << "Drawing track [" << i << "]" << " with depth gt = " << depth_mat.at<float>(ly,lx)
                << ", triang = (" << f << " * " << baseline << ") / " << lx - rightf.prev_.x_
                << " = " << depth << ", error: " << depth_error << "\n";
      cv::Mat lp_draw = lp_img.clone();
      cv::Mat rp_draw = rp_img.clone();
      cv::Mat lc_draw = lc_img.clone();
      cv::Mat rc_draw = rc_img.clone();
      DrawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, 4, color1);
      waitKey(0);
    }
    //if(depth_error > DEPTH_ERR_THR) {
    //  std::cout << "[EvalHelper]: warning - skippking big depth error!\n";
    //  continue;
    //}
    error_sum += depth_error;
    track_cnt++;
  }
  return error_sum / track_cnt;
}

double EvalHelper::getStereoRefinerDepthError(track::StereoTrackerBase& tracker_base,
                                              track::StereoTrackerBase& tracker,
                                              const double (&cam_params)[5], const cv::Mat& depth_mat,
                                              const cv::Mat lp_img, const cv::Mat rp_img,
                                              const cv::Mat lc_img, const cv::Mat rc_img)
{
  cv::Scalar color1(0,0,255);
  cv::Scalar color2(0,255,0);
  double f = cam_params[0];
  double baseline = cam_params[4];
  double error_sum = 0.0;
  int track_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo leftf = tracker.featureLeft(i);
    track::FeatureInfo rightf = tracker.featureRight(i);
    if(leftf.age_ <= 0) continue;
    double depth = triangulateDepth(leftf.prev_, rightf.prev_, f, baseline);
    // take the depth gt from nearest pixel
    int col = (int)std::round(leftf.prev_.x_);
    int row = (int)std::round(leftf.prev_.y_);

    double depth_error = std::abs(depth - depth_mat.at<float>(row, col));
    if(depth_error > DRAW_ERROR_THR) {
      continue;
      std::cout << "Drawing track [" << i << "]" << " with depth gt = " << depth_mat.at<float>(row,col)
                << ", triang = (" << f << " * " << baseline << ") / " << leftf.prev_.x_ - rightf.prev_.x_
                << " = " << depth << ", error: " << depth_error << "\n";
      //cv::Mat lp_draw = lp_img.clone();
      cv::Mat lp_draw, rp_draw, lc_draw, rc_draw;
      cv::cvtColor(lp_img, lp_draw, cv::COLOR_GRAY2RGB);
      cv::cvtColor(rp_img, rp_draw, cv::COLOR_GRAY2RGB);
      cv::cvtColor(lc_img, lc_draw, cv::COLOR_GRAY2RGB);
      cv::cvtColor(rc_img, rc_draw, cv::COLOR_GRAY2RGB);
      DrawStereoTrack(tracker, i, lp_draw, rp_draw, lc_draw, rc_draw, 4, color1);
      DrawStereoTrack(tracker_base, i, lp_draw, rp_draw, lc_draw, rc_draw, 4, color2);
      waitKey(0);
    }
    //if(depth_error > DEPTH_ERR_THR) {
    //  std::cout << "[EvalHelper]: warning - skippking big depth error!\n";
    //  continue;
    //}
    error_sum += depth_error;
    track_cnt++;
  }
  return error_sum / track_cnt;
}

void EvalHelper::drawStereoTrack(const core::Point& pt_l, const core::Point& pt_r,
                                 cv::Mat& img_l, cv::Mat& img_r)
{
  //double font_size = 0.4; // 0.3
  cv::Scalar color(255,0,0);
  cv::Point pt;

  pt.x = pt_l.x_;
  pt.y = pt_l.y_;
  cv::circle(img_l, pt, 2, color, -1, 8);
  //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
  //cv::putText(img_lc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

  pt.x = pt_r.x_;
  pt.y = pt_r.y_;
  cv::circle(img_r, pt, 2, color, -1, 8);
  //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
  //cv::putText(img_rc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

  cv::imshow("left_track", img_l);
  cv::imshow("right_track", img_r);
  waitKey(0);
}


double EvalHelper::getStereoReprojError(std::vector<core::Point>& points_lp, std::vector<core::Point>& points_rp,
    std::vector<core::Point>& points_lc, std::vector<core::Point>& points_rc,
    cv::Mat& C, cv::Mat& Rt, double baseline)
{
  assert(points_lp.size() == points_rp.size());
  assert(points_lp.size() == points_lc.size());
  assert(points_lc.size() == points_rc.size());

  // first triangulate image points in previous frame into 3D world points
  vector<Mat> prev_pts3d;
  prev_pts3d.resize(points_lp.size()); // - this cant work
  double f = C.at<double>(0,0);
  double cu = C.at<double>(0,2);
  double cv = C.at<double>(1,2);
  for(size_t i = 0; i < prev_pts3d.size(); i++) {
    //cout << "-1:\n" << points_lp[i] << endl;
    //double disp = max(points_lp[i].x_ - points_rp[i].x_, 0.01);
    double disp = max(points_lp[i].x_ - points_rp[i].x_, 0.0001);
    prev_pts3d[i] = Mat::zeros(4, 1, CV_64F);
    prev_pts3d[i].at<double>(0,0) = (points_lp[i].x_ - cu) * baseline / disp;
    prev_pts3d[i].at<double>(1,0) = (points_lp[i].y_ - cv) * baseline / disp;
    prev_pts3d[i].at<double>(2,0) = f * baseline / disp;
    prev_pts3d[i].at<double>(3,0) = 1.0;
    //cout << prev_pts3d[i] << endl;
  }

  // then evaluate the reprojection error across all points
  Mat curr_pts3d = Mat::zeros(4, 1, CV_64F);
  Mat img_pt = Mat::zeros(3, 1, CV_64F);
  Mat mat_pt2d = Mat::zeros(3, 1, CV_64F);
  double error = 0.0;
  double error_norm;
  for(size_t i = 0; i < prev_pts3d.size(); i++) {
    // transform 3d point from previous to current frame and project it to image plane
    //cout << i << ":  " << prev_pts3d[i] << endl;
    curr_pts3d = Rt * prev_pts3d[i];
    // for left camera
    img_pt = C * curr_pts3d;
    //cout << "1:\n" << img_pt << endl;
    img_pt /= img_pt.at<double>(2,0);
    //cout << img_pt << endl;

    point_to_cvmat(points_lc[i], mat_pt2d);
    error_norm = cv::norm(mat_pt2d - img_pt);
    error += error_norm * error_norm;

    // for right camera
    curr_pts3d.at<double>(0,0) = curr_pts3d.at<double>(0,0) - baseline;
    img_pt = C * curr_pts3d;
    img_pt /= img_pt.at<double>(2,0);
    //cout << img_pt << endl;
    point_to_cvmat(points_rc[i], mat_pt2d);
    error_norm = cv::norm(mat_pt2d - img_pt);
    error += error_norm * error_norm;
  }
  //return error / prev_pts3d.size();
  return error;
}

}
