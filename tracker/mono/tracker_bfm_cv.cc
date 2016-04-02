#include "tracker_bfm_cv.h"

#include "../base/helper_opencv.h"
#include "../stereo/debug_helper.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace
{
void DrawFeature(core::Point& pt, cv::Scalar& color, cv::Mat& img)
{
  cv::Point cvpt;
  cvpt.x = pt.x_;
  cvpt.y = pt.y_;
  cv::circle(img, cvpt, 1, color, -1, 8);
}
}

namespace track {

TrackerBFMcv::TrackerBFMcv(FeatureDetectorBase& detector, int max_features, int window_size, double max_distance) :
  detector_(detector), max_feats_(max_features), max_distance_(max_distance)
{
  matches_p_.resize(max_feats_);
  matches_c_.resize(max_feats_);
  desc_prev_ = cv::Mat::zeros(max_feats_, 64, CV_8U);
  desc_curr_ = cv::Mat::zeros(max_feats_, 64, CV_8U);
  age_.assign(max_feats_, -1);

  wsize_ = window_size;
  max_dist_x_ = window_size / 2;
  max_dist_y_ = window_size / 5;
}

//TrackerBFMcv::TrackerBFMcv(FeatureDetectorBase& detector, int max_features,
//                           int ws_left, int ws_right, int ws_up, int ws_down, double max_distance) :
//                           detector_(detector), max_feats_(max_features), max_distance_(max_distance)
//{
//  matches_p_.resize(max_feats_);
//  matches_c_.resize(max_feats_);
//  desc_prev_ = cv::Mat::zeros(max_feats_, 64, CV_8U);
//  desc_curr_ = cv::Mat::zeros(max_feats_, 64, CV_8U);
//  age_.assign(max_feats_, -1);
//
//  wsize_left_ = ws_left;
//  wsize_right_ = ws_right;
//  wsize_up_ = ws_up;
//  wsize_down_ = ws_down;
//  wsize_ = ws_left*2 + 1;
//}

int TrackerBFMcv::init(const cv::Mat& img)
{
  cvimg_c_ = img;

  // detect new features
  std::vector<cv::KeyPoint> feats;
  cv::Mat desc;
  detector_.detect(cvimg_c_, feats, desc);
  std::cout << "size: " << feats.size() << "\n";
  //cv::Mat disp_img;
  //cv::drawKeypoints(cvimg_c_, feats, disp_img);
  //cv::imshow("disp", disp_img);
  //cv::waitKey(0);
  //std::cout << desc;
  // init them
  for(size_t i = 0; i < age_.size(); i++) age_[i] = -1;
  prev_unused_feats_ = feats;
  prev_unused_desc_ = desc;

  //for(size_t i = 0; (i < feats.size() && i < matches_c_.size()); i++) {
  //  matches_c_[i] = feats[i];
  //  desc.row(i).copyTo(desc_curr_.row(i));
  //  age_[i] = -1;
  //}

  return 0;
}

int TrackerBFMcv::track(const cv::Mat& img)
{
  // age = -1 if feature is dead
  // age = 0 if feature is added just now (only curr matters)
  cvimg_p_ = cvimg_c_;
  cvimg_c_ = img;

  matches_p_ = matches_c_;
  desc_prev_ = desc_curr_.clone();

  cv::Mat desc;
  std::vector<cv::KeyPoint> feats;
  detector_.detect(cvimg_c_, feats, desc);
  std::vector<int> match_index;
  // match temporal
  std::cout << "[mono] matching prev-curr\n";
  match_features(matches_p_, feats, desc_prev_, desc, match_index, age_, false);

  std::vector<bool> unused_features;
  update_alive_tracks(feats, desc, match_index, unused_features);
  // remmeber all unused feats from before
  replace_dead_tracks(feats, desc, unused_features);
  save_unused_features(feats, desc, unused_features);
  return 0;
}

void TrackerBFMcv::match_features(const std::vector<cv::KeyPoint>& feats1,
                                  const std::vector<cv::KeyPoint>& feats2,
                                  const cv::Mat& desc1, const cv::Mat& desc2,
                                  std::vector<int>& match_index,
                                  const std::vector<int>& ages, bool replacing_dead)
{
  match_index.assign(feats1.size(), -1);
  //std::vector<int> matches_1to2, matches_2to1;
  //matches_1to2.resize(feats1.size());
  //matches_2to1.resize(feats2.size());
  std::vector<double> distances;
  distances.resize(feats1.size());

  // match 1 to 2
  #pragma omp parallel for
  for(size_t i = 0; i < feats1.size(); i++) {
    double dist, dx, dy;
    // dont track if the reference feature is dead
    if(!replacing_dead) {
      if(ages[i] < 0) {
        match_index[i] = -1;
        //matches_1to2[i] = -1;
        continue;
      }
    }

    int ind_best = -1;
    double dist_best = std::numeric_limits<double>::max();
    for(size_t j = 0; j < feats2.size(); j++) {
      dy = std::abs(feats1[i].pt.y - feats2[j].pt.y);
      dx = std::abs(feats1[i].pt.x - feats2[j].pt.x);
      // ignore features outside search area
      if(dx > max_dist_x_ || dy > max_dist_y_) continue;

      //cout << "match 1-2: " << i << " - " << j << endl;
      // TODO optimization: compare with SSE in blocks until the threshold is reached and then reject...
      dist = detector_.compare(desc1.row(i), desc2.row(j));
      //std::cout << "Distance = " << dist << "\n";

      if(dist < dist_best) {
        dist_best = dist;
        ind_best = j;
      }
    }

    distances[i] = dist_best;
    //std::cout << dist_best << " -- dist best\n";
    if(dist_best <= max_distance_)
      match_index[i] = ind_best;
      //matches_1to2[i] = ind_best;
    else
      match_index[i] = -1;
      //matches_1to2[i] = -1;
  }

  // match 2 to 1
  //for(size_t i = 0; i < feats2.size(); i++) {
  //  int ind_best = -1;
  //  double dist_best = std::numeric_limits<double>::max();
  //  for(size_t j = 0; j < feats1.size(); j++) {
  //    // dont track if the temporal reference feature is dead
  //    if(ages[j] < 0) {
  //      matches_2to1[i] = -1;
  //      continue;
  //    }
  //    dy = feats1[j].pt.y - feats2[i].pt.y;
  //    dx = feats1[j].pt.x - feats2[i].pt.x;
  //    if(dy < 0.0 && dy < -dyd) continue;
  //    if(dy > 0.0 && dy > dyu) continue;
  //    if(dx < 0.0 && dx < -dxr) continue;
  //    if(dx > 0.0 && dx > dxl) continue;

  //    // TODO - we can optimize this and put corrs in a map during the first match
  //    //cout << "match 2-1: " << i << " - " << j << endl;
  //    //corr = abs(getCorrelation(patches1[j], patches2[i]));
  //    dist = detector_.compare(desc1.row(j), desc2.row(i));
  //    if(dist < dist_best) {
  //      dist_best = dist;
  //      ind_best = j;
  //    }
  //    //cout << corr << endl;
  //  }
  //  //cout << corr_best << endl;
  //  if(dist_best <= max_distance_) {
  //    matches_2to1[i] = ind_best;
  //  }
  //  else
  //    matches_2to1[i] = -1;
  //}

  //// filter only the married features
  //for(int i = 0; i < feats1.size(); i++) {
  //  int m_1to2 = matches_1to2[i];
  //  // if two features were matced to each other then accept the match
  //  if(m_1to2 >= 0) {
  //    if(matches_2to1[m_1to2] == i)
  //      match_index[i] = m_1to2;
  //  }
  //}
}

// update all successfuly matched alive tracks
void TrackerBFMcv::update_alive_tracks(const std::vector<cv::KeyPoint>& feats,
                                       const cv::Mat& desc, const std::vector<int>& match_index,
                                       std::vector<bool>& unused_features)
{
  assert(match_index.size() == (size_t)max_feats_);
  unused_features.assign(feats.size(), true);
  int num_match = 0;
  for(size_t i = 0; i < matches_c_.size(); i++) {
    // if it is already dead from before, skip it so it can be replaced
    if(age_[i] < 0)
      continue;
    int mi = match_index[i];
    // check for match
    if(mi >= 0) {
      matches_c_[i] = feats[mi];
      desc.row(mi).copyTo(desc_curr_.row(i));
      age_[i]++;
      unused_features[mi] = false;
      num_match++;
      continue;
    }
    // else we have a dead feature :(
    death_count_++;
    age_acc_ += age_[i];
    age_[i] = -1;
  }
}

// use unused featurs from previous frame to find new matches to replace the dead ones
void TrackerBFMcv::replace_dead_tracks(const std::vector<cv::KeyPoint>& feats_c, const cv::Mat& desc,
                                      std::vector<bool>& unused_features)
{
  std::vector<int> match_index;
  match_features(prev_unused_feats_, feats_c, prev_unused_desc_, desc, match_index, age_, true);

  size_t j = 0;
  for(size_t i = 0; i < age_.size(); i++) {
    if(age_[i] < 0) {
      // first next matched feature
      if(j == match_index.size()) break;
      while(j < match_index.size()) {
        int mi = match_index[j];
        if(mi >= 0) {
          unused_features[mi] = false;
          // replace the dead one with unused new feature
          matches_p_[i] = prev_unused_feats_[j];
          matches_c_[i] = feats_c[mi];
          prev_unused_desc_.row(j).copyTo(desc_prev_.row(i));
          desc.row(mi).copyTo(desc_curr_.row(i));
          age_[i] = 1;
          j++;
          break;
        }
        j++;
      }
    }
  }
  std::cout << "number of final matches: " << countTracked() << " / " << max_feats_ << "\n";
}

// here we save all the unused current features to be used to replace dead feats in the next frame
void TrackerBFMcv::save_unused_features(const std::vector<cv::KeyPoint>& feats, const cv::Mat& desc,
                                        std::vector<bool>& unused_features)

{
  prev_unused_feats_.clear();
  //prev_unused_desc_ = cv::Mat(0, desc.cols, desc.type());
  prev_unused_desc_ = cv::Mat();
  // good thing the best Harris corners are from begining to end
  for(size_t i = 0; i < unused_features.size(); i++) {
    if(unused_features[i] == true) {
      prev_unused_feats_.push_back(feats[i]);
      prev_unused_desc_.push_back(desc.row(i).clone());
      //std::cout << prev_unused_desc_ << "\n\n";
      //cv::waitKey(0);
    }
  }
}

//void TrackerBFMcv::replaceDeadFeatures(const std::vector<cv::KeyPoint>& feats_c, const cv::Mat& desc,
//                                       std::vector<bool>& unused_features)
//{
//  size_t j = 0;
//  for(size_t i = 0; i < age_.size(); i++) {
//    if(age_[i] < 0) {
//      // first find next unused feature in matches set
//      while(j < unused_features.size()) {
//        if(unused_features[j] == true) {
//          unused_features[j] = false;
//          break;
//        }
//        j++;
//      }
//      // if no more unused feats
//      if(j >= unused_features.size())
//        break;
//      // replace the dead one with unused new feature
//      matches_c_[i] = feats_c[j];
//      desc.row(j).copyTo(desc_curr_.row(i));
//      age_[i] = 0;
//    }
//  }
//}

int TrackerBFMcv::countFeatures()
{
  return matches_p_.size();
}

int TrackerBFMcv::countTracked()
{
  int cnt = 0;
  for(size_t i = 0; i < age_.size(); i++) {
    if(age_[i] > 0)
      cnt++;
  }
  return cnt;
}

FeatureData TrackerBFMcv::getFeatureData(int i)
{
  FeatureData fdata;
  fdata.feat_ = feature(i);
  //fdata.desc_prev_ = desc_prev_.row(i).reshape(1, cbh_).clone();
  //fdata.desc_curr_ = desc_curr_.row(i).reshape(1, cbh_).clone();
  desc_prev_.row(i).reshape(1, wsize_).copyTo(fdata.desc_prev_);
  desc_curr_.row(i).reshape(1, wsize_).copyTo(fdata.desc_curr_);
  return fdata;
}

void TrackerBFMcv::printStats()
{
  std::cout << "[TrackerBFM] Active tracks: " << countTracked() << "\n";
  std::cout << "[TrackerBFM] Average track age: " << (double) age_acc_ / death_count_ << "\n";
}

FeatureInfo TrackerBFMcv::feature(int i)
{
  FeatureInfo feat;
  HelperOpencv::Keypoint2Point(matches_p_[i], feat.prev_);
  HelperOpencv::Keypoint2Point(matches_c_[i], feat.curr_);
  //feat.prev_ = matches_p_[i];
  //feat.curr_ = matches_c_[i];
  feat.age_ = age_[i];
  //feat.status_ = age_[i] + 1;
  return std::move(feat);
}


void TrackerBFMcv::showTrack(int i)
{
  cv::Mat img_p, img_c;
  cv::cvtColor(cvimg_p_, img_p, cv::COLOR_GRAY2RGB);
  cv::cvtColor(cvimg_c_, img_c, cv::COLOR_GRAY2RGB);
  FeatureInfo feat = feature(i);
  if(feat.age_ <= 0) throw "Error\n";
  cv::Scalar color(0,255,0);
  DrawFeature(feat.prev_, color, img_p);
  DrawFeature(feat.curr_, color, img_c);
  
  cv::resize(img_p, img_p, cv::Size(), 2.0, 2.0);
  cv::resize(img_c, img_c, cv::Size(), 2.0, 2.0);
  cv::imshow("prev", img_p);
  cv::imshow("curr", img_c);
  cv::waitKey(0);
}

} // end namespace
