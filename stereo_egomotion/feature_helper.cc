#include "feature_helper.h"

#include "opencv2/highgui/highgui.hpp"

#include "../tracker/base/eval_helper.h"

namespace vo {

using namespace core;
using namespace std;
using namespace track;
using namespace libviso;

void FeatureHelper::FilterRansacOutliers(track::StereoTrackerBase& tracker,
                                         const std::vector<int>& active_tracks,
                                         const std::vector<int>& inliers)
{
  std::vector<bool> dead_tracks(active_tracks.size(), true);
  for(size_t i = 0; i < inliers.size(); i++) {
    assert(inliers[i] >= 0 && inliers[i] < active_tracks.size());
    dead_tracks[inliers[i]] = false;
  }
  for(size_t i = 0; i < dead_tracks.size(); i++) {
    if(dead_tracks[i] == true)
      tracker.removeTrack(active_tracks[i]);
  }
}

//void FeatureHelper::FilterTracksWithPrior(track::StereoTrackerBase& tracker, const double* cam_params,
//                                          double max_z_dist, double min_disp)
//{
//  int filtered_cnt = 0;
//  for(int i = 0; i < tracker.countFeatures(); i++) {
//    track::FeatureInfo left = tracker.featureLeft(i);
//    track::FeatureInfo right = tracker.featureRight(i);
//    if(left.age_ < 1) continue;
//    double d_p = left.prev_.x_ - right.prev_.x_;
//    double d_c = left.curr_.x_ - right.curr_.x_;
//    //double max_disp_diff = 30.0; // 07
//    //double min_disp = 0.1;
//    //if(d_p < min_disp || d_c < min_disp)
//    //  printf("\33[0;31m Small disp: %f -> %f !\33[0m\n", d_p, d_c);
//    //if(std::abs(d_p - d_c) > max_disp_diff) {
//    if(std::abs(d_p - d_c) > max_disp_diff || d_p < min_disp || d_c < min_disp) {
//      //printf("\33[0;31m [Filtering]: Small disp or big disp difference: %f -> %f !\33[0m\n", d_p, d_c);
//      //std::cout << "Feature index = " << i << '\n';
//      //std::cout << "Previous:\n" << left.prev_ << '\n' << right.prev_ << '\n';
//      //std::cout << "Current:\n" << left.curr_ << '\n' << right.curr_ << '\n';
//      //tracker.showTrack(tracker_idx);
//
//      tracker.removeTrack(i);
//      filtered_cnt++;
//    }
//  }
//  printf("\33[0;31m [Filtering]: Num of filtered features = %d\33[0m\n", filtered_cnt);
//}

void FeatureHelper::FilterTracksWithPriorOld(track::StereoTrackerBase& tracker,
                                             double max_disp_diff, double min_disp)
{
  int filtered_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo left = tracker.featureLeft(i);
    track::FeatureInfo right = tracker.featureRight(i);
    if(left.age_ < 1) continue;
    double d_p = left.prev_.x_ - right.prev_.x_;
    double d_c = left.curr_.x_ - right.curr_.x_;
    //double max_disp_diff = 30.0; // 07
    //double min_disp = 0.1;
    //if(d_p < min_disp || d_c < min_disp)
    //  printf("\33[0;31m Small disp: %f -> %f !\33[0m\n", d_p, d_c);
    //if(std::abs(d_p - d_c) > max_disp_diff) {
    if(std::abs(d_p - d_c) > max_disp_diff || d_p < min_disp || d_c < min_disp) {
      //printf("\33[0;31m [Filtering]: Small disp or big disp difference: %f -> %f !\33[0m\n", d_p, d_c);
      //std::cout << "Feature index = " << i << '\n';
      //std::cout << "Previous:\n" << left.prev_ << '\n' << right.prev_ << '\n';
      //std::cout << "Current:\n" << left.curr_ << '\n' << right.curr_ << '\n';
      //tracker.showTrack(tracker_idx);

      tracker.removeTrack(i);
      filtered_cnt++;
    }
  }
  printf("\33[0;31m [Filtering]: Num of filtered features = %d\33[0m\n", filtered_cnt);
}

void FeatureHelper::FilterRansacOutliersWithPrior(track::StereoTrackerBase& tracker,
                                                  const std::vector<int>& active_tracks,
                                                  const std::vector<int>& inliers,
                                                  double max_disp_diff, double min_disp)
{
  std::vector<bool> dead_tracks(active_tracks.size(), true);
  for(size_t i = 0; i < inliers.size(); i++) {
    assert(inliers[i] >= 0 && inliers[i] < active_tracks.size());
    dead_tracks[inliers[i]] = false;
    int tracker_idx = active_tracks[inliers[i]];
    track::FeatureInfo left = tracker.featureLeft(tracker_idx);
    track::FeatureInfo right = tracker.featureRight(tracker_idx);
    double d_p = left.prev_.x_ - right.prev_.x_;
    double d_c = left.curr_.x_ - right.curr_.x_;
    //double max_disp_diff = 30.0; // 07
    //double min_disp = 0.1;
    //if(d_p < min_disp || d_c < min_disp)
    //  printf("\33[0;31m Small disp: %f -> %f !\33[0m\n", d_p, d_c);
    if(std::abs(d_p - d_c) > max_disp_diff || d_p < min_disp || d_c < min_disp) {
    //if(std::abs(d_p - d_c) > max_disp_diff) {
      printf("\33[0;31m Small disp or big disp difference: %f -> %f !\33[0m\n", d_p, d_c);
      std::cout << "Feature index = " << tracker_idx << '\n';
      std::cout << "Previous:\n" << left.prev_ << '\n' << right.prev_ << '\n';
      std::cout << "Current:\n" << left.curr_ << '\n' << right.curr_ << '\n';
      //tracker.showTrack(tracker_idx);
      dead_tracks[inliers[i]] = true;
    }
  }
  for(size_t i = 0; i < dead_tracks.size(); i++) {
    if(dead_tracks[i] == true)
      tracker.removeTrack(active_tracks[i]);
  }
}



void FeatureHelper::TrackerBaseToLibviso(StereoTrackerBase* tracker, std::vector<Matcher::p_match>& matches,
                                         std::vector<int>& active_tracks)
{
  matches.clear();
  active_tracks.clear();
  size_t feats_num = tracker->countFeatures();
  Matcher::p_match match;
  for(size_t i = 0; i < feats_num; i++) {
    FeatureInfo feat_left = tracker->featureLeft(i);
    FeatureInfo feat_right = tracker->featureRight(i);
    assert(feat_left.age_ == feat_right.age_);
    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      match.u1p = feat_left.prev_.x_;
      match.v1p = feat_left.prev_.y_;
      match.u1c = feat_left.curr_.x_;
      match.v1c = feat_left.curr_.y_;

      match.u2p = feat_right.prev_.x_;
      match.v2p = feat_right.prev_.y_;
      match.u2c = feat_right.curr_.x_;
      match.v2c = feat_right.curr_.y_;

      if(std::isnan(match.u1p) || std::isnan(match.v1p) || std::isnan(match.u1c) || std::isnan(match.v1c)
         || std::isnan(match.u2p) || std::isnan(match.v2p) || std::isnan(match.u2c) || std::isnan(match.v2c))
        throw "NAN!";

      matches.push_back(match);
      active_tracks.push_back(i);
    }
  }
}

void FeatureHelper::LibvisoToTrackerBase(std::vector<Matcher::p_match>& matches,
                                         std::vector<FeatureInfo>& feats_left,
                                         std::vector<FeatureInfo>& feats_right)
{
  feats_left.clear();
  feats_right.clear();

  FeatureInfo feat_left;
  FeatureInfo feat_right;

  for(size_t i = 0; i < matches.size(); i++) {
    feat_left.status_ = 0;
    feat_left.age_ = 1;
    feat_left.prev_.x_ = matches.at(i).u1p;
    feat_left.prev_.y_ = matches.at(i).v1p;
    feat_left.curr_.x_ = matches.at(i).u1c;
    feat_left.curr_.y_ = matches.at(i).v1c;
    feat_right.status_ = 0;
    feat_right.age_ = 1;
    feat_right.prev_.x_ = matches.at(i).u2p;
    feat_right.prev_.y_ = matches.at(i).v2p;
    feat_right.curr_.x_ = matches.at(i).u2c;
    feat_right.curr_.y_ = matches.at(i).v2c;

    feats_left.push_back(feat_left);
    feats_right.push_back(feat_right);
  }
}

//void FeatureHelper::filterOutlierTracks(track::StereoTrackerBase& tracker, const std::vector<int>& active_tracks,
//                                        const std::vector<int>& inliers, std::vector<int>& outliers)
//{
//  outliers.clear();
//  std::vector<bool> dead_tracks(active_tracks.size(), true);
//  for(size_t i = 0; i < inliers.size(); i++) {
//    assert(inliers[i] >= 0 && inliers[i] < active_tracks.size());
//    dead_tracks[inliers[i]] = false;
//    track::FeatureInfo left = tracker.featureLeft(active_tracks[inliers[i]]);
//    track::FeatureInfo right = tracker.featureRight(active_tracks[inliers[i]]);
//    double d_p = left.prev_.x_ - right.prev_.x_;
//    double d_c = left.curr_.x_ - right.curr_.x_;
//    if(std::abs(d_p - d_c) > 10.0)
//      printf("\33[0;31m BIG disp difference: %f -> %f !\33[0m\n", d_p, d_c);
//  }
//  for(size_t i = 0; i < dead_tracks.size(); i++) {
//    if(dead_tracks[i] == true) {
//      tracker.removeTrack(active_tracks[i]);
//      outliers.push_back(active_tracks[i]);
//    }
//  }
//}


void FeatureHelper::TrackerBaseToLibviso(StereoTrackerBase* tracker, std::vector<Matcher::p_match>& matches)
{
  matches.clear();

  size_t feats_num = tracker->countFeatures();

  Matcher::p_match match;
  for(size_t i = 0; i < feats_num; i++) {
    FeatureInfo feat_left = tracker->featureLeft(i);
    FeatureInfo feat_right = tracker->featureRight(i);
    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      //cout << feat_left.prev_ << " -> " << feat_left.curr_ << "\n";
      //cout << feat_right.prev_ << " -> " << feat_right.curr_ << "\n";
      match.u1p = feat_left.prev_.x_;
      match.v1p = feat_left.prev_.y_;
      match.u1c = feat_left.curr_.x_;
      match.v1c = feat_left.curr_.y_;

      match.u2p = feat_right.prev_.x_;
      match.v2p = feat_right.prev_.y_;
      match.u2c = feat_right.curr_.x_;
      match.v2c = feat_right.curr_.y_;

      matches.push_back(match);
    }
  }
}



void FeatureHelper::getActiveTracks(StereoTrackerBase* tracker, std::vector<int>& active_tracks)
{
  active_tracks.clear();
  size_t feats_num = tracker->countFeatures();
  Matcher::p_match match;
  for(size_t i = 0; i < feats_num; i++) {
    FeatureInfo feat_left = tracker->featureLeft(i);
    FeatureInfo feat_right = tracker->featureRight(i);
    assert(feat_left.age_ == feat_right.age_);
    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      active_tracks.push_back(i);
    }
  }
}

void FeatureHelper::TrackerBaseToLibvisoStratified(const track::StereoTrackerBase* tracker,
    std::vector<Matcher::p_match>& matches_strat, int max_features, double block_width, double block_height,
    core::Size img_size, std::vector<int>& active_tracks)
{
  matches_strat.clear();

  size_t feats_num = tracker->countFeatures();

  int cols = (img_size.width_ / block_width);
  int rows = (img_size.height_ / block_height);
  //cout << "Stratify with: " <<  rows << " -- " << cols << endl;

  double block_w = (double)img_size.width_ / cols;
  double block_h = (double)img_size.height_ / rows;

  vector<int> blocks;
  blocks.resize(rows * cols);
  std::fill(blocks.begin(), blocks.end(), 0);

  int r, c;
  Matcher::p_match match;
  for(size_t i = 0; i < feats_num; i++) {
    FeatureInfo feat_left = tracker->featureLeft(i);
    FeatureInfo feat_right = tracker->featureRight(i);

    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      //cout << feat_left.prev_ << " -> " << feat_left.curr_ << "\n";
      //cout << feat_right.prev_ << " -> " << feat_right.curr_ << "\n";

      c = (int)feat_left.prev_.x_ / block_w;
      r = (int)feat_left.prev_.y_ / block_h;

      //if(i == 151) {
      //   cout << feat_left.prev_ << " - " << "\n";
      //   cout << "idx=" << i << ":\n\tc = " << c << " / " << cols << "\n\tr = " << r << " / " << rows << "\n";
      //}
      assert(c >= 0 && c < cols);
      assert(r >= 0 && r < rows);

      if(blocks[r*cols+c] <= max_features) {
        match.u1p = feat_left.prev_.x_;
        match.v1p = feat_left.prev_.y_;
        match.u1c = feat_left.curr_.x_;
        match.v1c = feat_left.curr_.y_;

        match.u2p = feat_right.prev_.x_;
        match.v2p = feat_right.prev_.y_;
        match.u2c = feat_right.curr_.x_;
        match.v2c = feat_right.curr_.y_;

        matches_strat.push_back(match);
        blocks[r*cols+c]++;
        active_tracks.push_back(i);
      }
    }
  }
}

void FeatureHelper::TrackerBaseToLibvisoUniform(const track::StereoTrackerBase* tracker,
    std::vector<Matcher::p_match>& matches_strat, int max_features, int rows, int cols,
    core::Size img_size, std::vector<int>& active_tracks)
{
  matches_strat.clear();

  size_t feats_num = tracker->countFeatures();

  double block_w = (double)img_size.width_ / cols;
  double block_h = (double)img_size.height_ / rows;

  vector<int> blocks;
  blocks.resize(rows * cols);
  std::fill(blocks.begin(), blocks.end(), 0);

  int r, c;
  Matcher::p_match match;
  for(size_t i = 0; i < feats_num; i++) {
    FeatureInfo feat_left = tracker->featureLeft(i);
    FeatureInfo feat_right = tracker->featureRight(i);

    if(feat_left.age_ > 0 && feat_right.age_ > 0) {
      //cout << feat_left.prev_ << " -> " << feat_left.curr_ << "\n";
      //cout << feat_right.prev_ << " -> " << feat_right.curr_ << "\n";

      c = (int)feat_left.prev_.x_ / block_w;
      r = (int)feat_left.prev_.y_ / block_h;

      //if(i == 151) {
      //   cout << feat_left.prev_ << " - " << "\n";
      //   cout << "idx=" << i << ":\n\tc = " << c << " / " << cols << "\n\tr = " << r << " / " << rows << "\n";
      //}
      assert(c >= 0 && c < cols);
      assert(r >= 0 && r < rows);

      if(blocks[r*cols+c] <= max_features) {
        match.u1p = feat_left.prev_.x_;
        match.v1p = feat_left.prev_.y_;
        match.u1c = feat_left.curr_.x_;
        match.v1c = feat_left.curr_.y_;

        match.u2p = feat_right.prev_.x_;
        match.v2p = feat_right.prev_.y_;
        match.u2c = feat_right.curr_.x_;
        match.v2c = feat_right.curr_.y_;

        matches_strat.push_back(match);
        blocks[r*cols+c]++;
        active_tracks.push_back(i);
      }
    }
  }
}
void FeatureHelper::LibvisoInliersToPoints(vector<Matcher::p_match>& matches, vector<int>& inliers,
    vector<core::Point>& points_lp, vector<core::Point>& points_rp,
    vector<core::Point>& points_lc, vector<core::Point>& points_rc)
{
  size_t nInliers = inliers.size();
  points_lp.resize(nInliers);
  points_rp.resize(nInliers);
  points_lc.resize(nInliers);
  points_rc.resize(nInliers);
  core::Point pt;
  for(size_t i = 0; i < nInliers; i++) {
    //cout << "match " << i << endl;
    int idx = inliers[i];
    points_lp[i].x_ = matches[idx].u1p;
    points_lp[i].y_ = matches[idx].v1p;
    points_rp[i].x_ = matches[idx].u2p;
    points_rp[i].y_ = matches[idx].v2p;
    points_lc[i].x_ = matches[idx].u1c;
    points_lc[i].y_ = matches[idx].v1c;
    points_rc[i].x_ = matches[idx].u2c;
    points_rc[i].y_ = matches[idx].v2c;
  }
}

int FeatureHelper::filterBadTracks(track::StereoTrackerBase& tracker)
{
  double thr_disp = 0.001;
  int bad_cnt = 0;
  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo left = tracker.featureLeft(i);
    track::FeatureInfo right = tracker.featureRight(i);
    // skip if not active track
    if(left.age_ < 1)
      continue;
    double disp = left.prev_.x_ - right.prev_.x_;
    if(disp <= thr_disp) {
      tracker.removeTrack(i);
      bad_cnt++;
      continue;
    }
    disp = left.curr_.x_ - right.curr_.x_;
    if(disp <= thr_disp) {
      tracker.removeTrack(i);
      bad_cnt++;
    }
  }
  return bad_cnt;
}

void FeatureHelper::filterOutlierTracks(track::StereoTrackerBase& tracker, const cv::Mat& Rt,
                                        const double (&cam_params)[5], std::vector<int>& outliers,
                                        const double eps)
{
  //std::vector<int> outliers;

  double f = cam_params[0];
  double cx = cam_params[2];
  double cy = cam_params[3];
  double b = cam_params[4];

  cv::Mat C = cv::Mat::zeros(3, 4, CV_64F);
  C.at<double>(0,0) = f;
  C.at<double>(1,1) = f;
  C.at<double>(2,2) = 1.0;
  C.at<double>(0,2) = cx;
  C.at<double>(1,2) = cy;

  double error_norm;
  cv::Mat pt3d = cv::Mat::zeros(4, 1, CV_64F);
  cv::Mat mat_pt2d = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat img_pt = cv::Mat::zeros(3, 1, CV_64F);

  for(int i = 0; i < tracker.countFeatures(); i++) {
    track::FeatureInfo left = tracker.featureLeft(i);
    track::FeatureInfo right = tracker.featureRight(i);

    // skip if not active track
    if(left.age_ < 1)
      continue;

    double disp = std::max(left.prev_.x_ - right.prev_.x_, 0.0001);
    // triangulate, transform and reproject
    pt3d.at<double>(0) = (left.prev_.x_ - cx) * b / disp;
    pt3d.at<double>(1) = (left.prev_.y_ - cy) * b / disp;
    pt3d.at<double>(2) = f * b / disp;
    pt3d.at<double>(3) = 1.0;
    pt3d = Rt * pt3d;

    // calculate reprojection error
    // for left camera
    img_pt = C * pt3d;
    //cout << "1:\n" << img_pt << endl;
    img_pt /= img_pt.at<double>(2,0);
    //cout << img_pt << endl;

    EvalHelper::voPoint2cvMat(left.curr_, mat_pt2d);
    error_norm = cv::norm(mat_pt2d - img_pt);

    // for right camera
    pt3d.at<double>(0,0) = pt3d.at<double>(0,0) - b;
    img_pt = C * pt3d;
    img_pt /= img_pt.at<double>(2,0);
    //cout << img_pt << endl;
    EvalHelper::voPoint2cvMat(right.curr_, mat_pt2d);
    error_norm += cv::norm(mat_pt2d - img_pt);

    // remove if it is an outlier
    //std::cout << error_norm << endl;
    if(error_norm > eps) {
      tracker.removeTrack(i);
      outliers.push_back(i);
    }
  }
}

void FeatureHelper::drawStereoRefinerTracks(track::StereoTrackerBase& tracker,
                                            track::StereoTrackerBase& tracker_refiner,
                                            cv::Mat& img_lp, cv::Mat& img_rp)
{
  double font_size = 0.4; // 0.3
  cv::Point pt1, pt2;
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);
  cv::Scalar color_ref(0,255,0);
  for(int i = 0; i < tracker_refiner.countFeatures(); i++) {
    FeatureInfo feat_left = tracker.featureLeft(i);
    FeatureInfo feat_right = tracker.featureRight(i);
    FeatureInfo feat_left_ref = tracker_refiner.featureLeft(i);
    FeatureInfo feat_right_ref = tracker_refiner.featureRight(i);
    assert(feat_left.age_ == feat_left_ref.age_);

    if(feat_left.age_ > 0) {
      pt1.x = feat_left.prev_.x_;
      pt1.y = feat_left.prev_.y_;
      pt2.x = feat_left.curr_.x_;
      pt2.y = feat_left.curr_.y_;
      cv::line(img_lp, pt1, pt2, color_prev, 2, 8);
      cv::circle(img_lp, pt1, 2, color_prev, -1, 8);
      cv::circle(img_lp, pt2, 2, color_curr, -1, 8);
      pt1.x = feat_left_ref.prev_.x_;
      pt1.y = feat_left_ref.prev_.y_;
      pt2.x = feat_left_ref.curr_.x_;
      pt2.y = feat_left_ref.curr_.y_;
      cv::line(img_lp, pt1, pt2, color_ref, 1, 8);
      cv::circle(img_lp, pt1, 2, color_ref, -1, 8);
      cv::circle(img_lp, pt2, 2, color_ref, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      //cv::putText(img_lc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

      pt1.x = feat_right.prev_.x_;
      pt1.y = feat_right.prev_.y_;
      pt2.x = feat_right.curr_.x_;
      pt2.y = feat_right.curr_.y_;
      cv::line(img_rp, pt1, pt2, color_prev, 2, 8);
      cv::circle(img_rp, pt1, 2, color_prev, -1, 8);
      cv::circle(img_rp, pt2, 2, color_curr, -1, 8);
      pt1.x = feat_right_ref.prev_.x_;
      pt1.y = feat_right_ref.prev_.y_;
      pt2.x = feat_right_ref.curr_.x_;
      pt2.y = feat_right_ref.curr_.y_;
      cv::line(img_rp, pt1, pt2, color_ref, 1, 8);
      cv::circle(img_rp, pt1, 2, color_ref, -1, 8);
      cv::circle(img_rp, pt2, 2, color_ref, -1, 8);
      //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
      //cv::putText(img_rc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);
    }

  }
  cv::Mat disp_lp, disp_rp;
  cv::resize(img_lp, disp_lp, cv::Size(), 2.0, 2.0);
  cv::resize(img_rp, disp_rp, cv::Size(), 2.0, 2.0);
  cv::imshow("left_prev_track", disp_lp);
  cv::imshow("right_prev_track", disp_rp);
  //cv::imshow("left_prev_track", img_lp);
  //cv::imshow("right_prev_track", img_rp);

  //vector<int> compression_params;
  //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  //compression_params.push_back(9);
  //cv::imwrite("tracks_left.png", img_lc, compression_params);
  //cv::imwrite("tracks_right.png", img_rc, compression_params);
  //cv::waitKey(0);
}


void FeatureHelper::drawFeatures(const std::vector<core::Point>& features, const cv::Mat& image)
{
  double font_size = 0.4; // 0.3
  cv::Point pt;
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);

  cv::Mat img;
  cvtColor(image, img, cv::COLOR_GRAY2RGB);

  for(int i = 0; i < features.size(); i++) {
    pt.x = features[i].x_;
    pt.y = features[i].y_;
    cv::circle(img, pt, 2, color_prev, -1, 8);
    cv::imshow("features", img);
    //cv::waitKey(0);
    //cv::putText(img_lc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);
  }
  //vector<int> compression_params;
  //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  //compression_params.push_back(9);
  //cv::imwrite("harris_corners.png", img, compression_params);
}


void FeatureHelper::drawStereoTracks(StereoTrackerBase& tracker, const std::vector<int>& tracks,
                                     cv::Mat& img_lc, cv::Mat& img_rc)
{
  double font_size = 0.4; // 0.3
  cv::Point pt1, pt2;
  cv::Scalar color_curr(0,0,255);
  cv::Scalar color_prev(255,0,0);
  for(int i = 0; i < tracks.size(); i++) {
    FeatureInfo feat_left = tracker.featureLeft(tracks[i]);
    FeatureInfo feat_right = tracker.featureRight(tracks[i]);
    //cout << feat_left.status_ << endl;
    pt1.x = feat_left.prev_.x_;
    pt1.y = feat_left.prev_.y_;
    pt2.x = feat_left.curr_.x_;
    pt2.y = feat_left.curr_.y_;
    cv::line(img_lc, pt1, pt2, color_prev, 2, 8);
    cv::circle(img_lc, pt1, 2, color_prev, -1, 8);
    cv::circle(img_lc, pt2, 2, color_curr, -1, 8);
    //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
    cv::putText(img_lc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);

    pt1.x = feat_right.prev_.x_;
    pt1.y = feat_right.prev_.y_;
    pt2.x = feat_right.curr_.x_;
    pt2.y = feat_right.curr_.y_;
    cv::line(img_rc, pt1, pt2, color_prev, 2, 8);
    cv::circle(img_rc, pt1, 2, color_prev, -1, 8);
    cv::circle(img_rc, pt2, 2, color_curr, -1, 8);
    //cout << pt1.x << ", " << pt1.y << " --> " << pt2.x << ", " << pt2.y << endl;
    cv::putText(img_rc, to_string(i), pt1, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar::all(0), 1, 8);
  }
  cv::imshow("left_prev_track_outliers", img_lc);
  cv::imshow("right_prev_track_outliers", img_rc);
}


}
