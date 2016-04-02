#include "tracker_bfm.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#define DEBUG_ON

namespace
{
void DrawFeature(const cv::Point2f& pt, const cv::Scalar& color, cv::Mat& img)
{
  cv::Point cvpt;
  cvpt.x = std::round(pt.x);
  cvpt.y = std::round(pt.y);
  cv::circle(img, pt, 1, color, -1, 8);
}

void DrawFeature(const core::Point& pt, const cv::Scalar& color, cv::Mat& img)
{
  cv::Point cvpt;
  cvpt.x = std::round(pt.x_);
  cvpt.y = std::round(pt.y_);
  cv::circle(img, cvpt, 1, color, -1, 8);
}

//void RenderPatch(const cv::Mat& patch, int rows, cv::Mat& img)
//{
//  cv::Size imgsz = cv::Size(200, 200);
//  img = patch.clone();
//  img = img.reshape(0, rows);
//  cv::resize(img, img, imgsz, 0, 0, cv::INTER_NEAREST);
//}

}

namespace track {


TrackerBFM::TrackerBFM(FeatureDetectorBase& detector, int max_features, double min_ncc,
                       int patch_size, int wsz, bool match_with_oldest) :
                              detector_(detector), max_feats_(max_features),
                              min_ncc_(min_ncc), match_with_oldest_(match_with_oldest)
{
  patch_size_ = patch_size;
  matches_p_.resize(max_feats_);
  matches_c_.resize(max_feats_);
  desc_ref_.resize(max_feats_);
  desc_curr_.resize(max_feats_);
  age_.assign(max_feats_, -1);

  wsize_ = wsz;
  // TODO
  //MAX dist x = 150
  //MAX dist y = 52
  max_dist_x_ = wsz / 2;
  max_dist_y_ = wsz / 5;
  //max_dist_y_ = wsz / 2;
}

int TrackerBFM::init(const cv::Mat& img)
{
  cvimg_c_ = img;

  // detect new features
  detector_.detect(cvimg_c_, prev_unused_feats_);
  std::vector<core::DescriptorNCC> desc;
  compute_ncc_descriptors(cvimg_c_, prev_unused_feats_, prev_unused_desc_);

  //cv::Mat disp_img;
  //cvimg_c_.convertTo(disp_img, CV_8U);
  //cv::drawKeypoints(disp_img, prev_unused_feats_, disp_img);
  //cv::imshow("disp", disp_img);
  //cv::waitKey(0);

  return 0;
}

int TrackerBFM::track(const cv::Mat& img)
{
  cvimg_p_ = cvimg_c_;
  cvimg_c_ = img;

  matches_p_ = matches_c_;
  if(!match_with_oldest_)
    desc_ref_ = desc_curr_;

  std::vector<cv::KeyPoint> feats;
  std::vector<core::DescriptorNCC> desc;
  detector_.detect(cvimg_c_, feats);
  compute_ncc_descriptors(cvimg_c_, feats, desc);
  std::vector<int> match_index;
  // match temporal
  //std::cout << "[TrackerBFM] matching prev-curr\n";
  //match_features(matches_p_, feats, desc_prev_, desc, match_index, half_wsize_, age_, false);
  match_features(matches_p_, feats, desc_ref_, desc, match_index, age_, false);

  std::vector<bool> unused_features;
  update_alive_tracks(feats, desc, match_index, unused_features);
  // remmeber all unused feats from before
  //std::cout << "[TrackerBFM] Replacing dead features\n";
  replace_dead_tracks(feats, desc, unused_features);
  std::cout << "[TrackerBFM]: Number of final matches: " << countTracked() << " / " << max_feats_ << "\n";
  save_unused_features(feats, desc, unused_features);
  return 0;
}

//void TrackerBFM::match_features(const std::vector<cv::KeyPoint>& feats1,
//                                const std::vector<cv::KeyPoint>& feats2,
//                                const std::vector<core::DescriptorNCC>& desc1,
//                                const std::vector<core::DescriptorNCC>& desc2,
//                                std::vector<int>& match_index,
//                                const std::vector<int>& ages, bool replacing_dead)
//{
//  match_index.assign(feats1.size(), -1);
//  //std::vector<int> matches_1to2, matches_2to1;
//  //matches_1to2.resize(feats1.size());
//  //matches_2to1.resize(feats2.size());
//  std::vector<double> distances;
//  distances.resize(feats1.size());
//
//  cv::Mat disp_1_track, disp_2_track;
//
//#ifndef DEBUG_ON
//  #pragma omp parallel for
//#endif
//  for(size_t i = 0; i < feats1.size(); i++) {
//    double dist, dx, dy;
//    // dont track if the reference feature is dead
//    // do this only if not repleacing dead
//    if(!replacing_dead) {
//      if(ages[i] < 0) {
//        match_index[i] = -1;
//        continue;
//      }
//    }
//
//    int ind_best = -1;
//    double dist_best = -1.0;
//#ifdef DEBUG_ON
//    bool debug = true;
//#endif
//    double dist_second_best = -1.0;
//    bool filter_ambiguous = false;
//    for(size_t j = 0; j < feats2.size(); j++) {
//      dy = std::abs(feats1[i].pt.y - feats2[j].pt.y);
//      dx = std::abs(feats1[i].pt.x - feats2[j].pt.x);
//      // ignore features outside search area
//      //if(dx > half_search_sz || dy > half_search_sz) continue;
//      if(dx > max_dist_x_ || dy > max_dist_y_) continue;
//      //cout << "match 1-2: " << i << " - " << j << endl;
//      // TODO optimization: compare with SSE in blocks until the threshold is reached and then reject...
//      //dist = detector_.compare(desc1.row(i), desc2.row(j));
//      dist = recon::StereoCosts::get_cost_NCC(desc1[i], desc2[j]);
//
//      //// debug
//      //printf("Distance = %f  (the best = %f)\n", dist, dist_best);
//      //cv::Scalar color(0,255,0);
//      //cv::Mat cimg_prev, cimg_curr;
//      //cv::cvtColor(cvimg_p_, cimg_prev, cv::COLOR_GRAY2RGB);
//      //cv::cvtColor(cvimg_c_, cimg_curr, cv::COLOR_GRAY2RGB);
//      //DrawFeature(feats1[i].pt, color, cimg_prev);
//      //DrawFeature(feats2[j].pt, color, cimg_curr);
//      //cv::imshow("mono_bfm_img_prev", cimg_prev);
//      //cv::imshow("mono_bfm_img_curr", cimg_curr);
//      //cv::waitKey(0);
//
//#ifdef DEBUG_ON
//      //if(debug && (i == 359)) {
//      if(debug) {
//        cv::Point2f pt;
//        cv::cvtColor(cvimg_p_, disp_1_track, cv::COLOR_GRAY2RGB);
//        pt.x = feats1[i].pt.x;
//        pt.y = feats1[i].pt.y;
//        cv::circle(disp_1_track, pt, 2, cv::Scalar(255,0,0), 2, 8);
//        cv::Point pt_rec1, pt_rec2;
//        cv::Rect rect;
//        rect.x = pt.x - max_dist_x_;
//        rect.y = pt.y - max_dist_y_;
//        rect.width = 2 * max_dist_x_ + 1;
//        rect.height = 2 * max_dist_y_ + 1;
//        //pt_rec1.x_ = pt.x - dxl;
//        //pt_rec1.y_ = pt.y - dyd;
//        //pt_rec2.x_ = pt.x + dxr;
//        //pt_rec2.y_ = pt.y + dyd;
//        rectangle(disp_1_track, rect, cv::Scalar(255,0,0), 1, 8);
//        //cout << "size: " << disp_left_track.size();
//        imshow("image_1", disp_1_track);
//
//        cv::Point2f pt1, pt2, pt_best;
//        std::cout << i << " - " << j << ":\n";
//        std::cout << feats1[i].pt << " - " << feats2[j].pt << "\n";
//        std::cout << "correlation: " << dist << "  (the best: " << dist_best << ")\n";
//        cv::cvtColor(cvimg_c_, disp_2_track, cv::COLOR_GRAY2RGB);
//        pt2.x = feats2[j].pt.x;
//        pt2.y = feats2[j].pt.y;
//        if(ind_best >= 0) {
//           pt_best.x = feats2[ind_best].pt.x;
//           pt_best.y = feats2[ind_best].pt.y;
//           cv::circle(disp_2_track, pt_best, 2, cv::Scalar(0,255,0), 2, 8);
//        }
//        //cout << pt1 << " <--> " << pt2 << "\n--------------------------------\n";
//        cv::circle(disp_2_track, pt2, 2, cv::Scalar(255,0,0), 2, 8);
//        cv::rectangle(disp_2_track, rect, cv::Scalar(255,0,0), 1, 8);
//        cv::imshow("image_2", disp_2_track);
//        //cv::Mat disp_patch_1, disp_patch_2, disp_patch_best;
//        //RenderPatch(desc1[i].vec, cbh_, disp_patch_1);
//        HelperOpencv::DrawFloatDescriptor(desc1[i].vec, patch_size_, "patch_1");
//        //RenderPatch(desc2[j].vec, cbh_, disp_patch_2);
//        HelperOpencv::DrawFloatDescriptor(desc2[j].vec, patch_size_, "patch_2");
//        //cout << patches1[i].mat_ << endl;
//        //cout << patches2[j].mat_ << endl;
//        //cv::imshow("patch_1", disp_patch_1);
//        //cv::imshow("patch_2", disp_patch_2);
//        if(ind_best >= 0)
//          HelperOpencv::DrawFloatDescriptor(desc2[ind_best].vec, patch_size_, "patch_2_best");
//          //RenderPatch(desc2[ind_best].vec, cbh_, disp_patch_best);
//          //cv::imshow("patch_2_best", disp_patch_best);
//        int key = cv::waitKey(0);
//        if(key == 27)
//          debug = false;
//      }
//#endif
//
//      if(dist > dist_best) {
//        if(filter_ambiguous)
//          dist_second_best = dist_best;
//        dist_best = dist;
//        ind_best = j;
//      }
//    }
//
//    distances[i] = dist_best;
//    //std::cout << dist_best << " -- dist best\n";
//    if(!filter_ambiguous && dist_best > min_ncc_)
//      match_index[i] = ind_best;
//    else if(filter_ambiguous && dist_best > min_ncc_ && dist_second_best < min_ncc_)
//      match_index[i] = ind_best;
//      //matches_1to2[i] = ind_best;
//    else
//      match_index[i] = -1;
//      //matches_1to2[i] = -1;
//  }
//}

void TrackerBFM::match_features(const std::vector<cv::KeyPoint>& feats1,
                                const std::vector<cv::KeyPoint>& feats2,
                                const std::vector<core::DescriptorNCC>& desc1,
                                const std::vector<core::DescriptorNCC>& desc2,
                                std::vector<int>& match_index,
                                const std::vector<int>& ages, bool replacing_dead)
{
  match_index.assign(feats1.size(), -1);
  std::vector<int> matches_1to2, matches_2to1;
  matches_1to2.assign(feats1.size(), -1);
  matches_2to1.assign(feats2.size(), -1);
  //std::vector<double> distances;
  //distances.resize(feats1.size());

  // match 1 to 2
#ifndef DEBUG_ON
  #pragma omp parallel for
#endif
  for(size_t i = 0; i < feats1.size(); i++) {
    double dist, dx, dy;
    // dont track if the reference feature is dead
    // do this only if not repleacing dead
    if(!replacing_dead) {
      if(ages[i] < 0) {
        matches_1to2[i] = -1;
        continue;
      }
    }

    int ind_best = -1;
    double dist_best = -1.0;
    //double dist_second_best = -1.0;
    for(size_t j = 0; j < feats2.size(); j++) {
      dy = std::abs(feats1[i].pt.y - feats2[j].pt.y);
      dx = std::abs(feats1[i].pt.x - feats2[j].pt.x);
      // ignore features outside search area
      if(dx > max_dist_x_ || dy > max_dist_y_) continue;

      // TODO optimization: compare with SSE in blocks until the threshold is reached and then reject...
      //dist = detector_.compare(desc1.row(i), desc2.row(j));
      dist = recon::StereoCosts::get_cost_NCC(desc1[i], desc2[j]);

      if(dist > dist_best) {
        dist_best = dist;
        ind_best = j;
      }
    }

    //distances[i] = dist_best;
    //std::cout << dist_best << " -- dist best\n";
    if(dist_best > min_ncc_)
      //match_index[i] = ind_best;
      matches_1to2[i] = ind_best;
    else
      //match_index[i] = -1;
      matches_1to2[i] = -1;
  }

  // match 2 to 1
#ifndef DEBUG_ON
  #pragma omp parallel for
#endif
  for(size_t i = 0; i < feats2.size(); i++) {
    double dist, dx, dy;

    int ind_best = -1;
    double dist_best = -1.0;
    //double dist_second_best = -1.0;
    for(size_t j = 0; j < feats1.size(); j++) {
      // dont track if the reference feature is dead
      // do this only if not repleacing dead
      if(!replacing_dead) {
        if(ages[j] < 0)
          continue;
      }

      dy = std::abs(feats1[j].pt.y - feats2[i].pt.y);
      dx = std::abs(feats1[j].pt.x - feats2[i].pt.x);
      // ignore features outside search area
      if(dx > max_dist_x_ || dy > max_dist_y_) continue;

      //cout << "match 1-2: " << i << " - " << j << endl;
      // TODO optimization: compare with SSE in blocks until the threshold is reached and then reject...
      //dist = detector_.compare(desc1.row(i), desc2.row(j));
      dist = recon::StereoCosts::get_cost_NCC(desc1[j], desc2[i]);

      if(dist > dist_best) {
        //dist_second_best = dist_best;
        dist_best = dist;
        ind_best = j;
      }
    }

    //distances[i] = dist_best;
    //std::cout << dist_best << " -- dist best\n";
    if(dist_best > min_ncc_)
      matches_2to1[i] = ind_best;
    else
      matches_2to1[i] = -1;
  }

  // filter only the married features
  for(int i = 0; i < (int)feats1.size(); i++) {
    int m_1to2 = matches_1to2[i];
    // if two features were matced to each other then accept the match
    if(m_1to2 >= 0) {
      if(matches_2to1[m_1to2] == i)
        match_index[i] = m_1to2;
    }
  }
}

// update all successfuly matched alive tracks
void TrackerBFM::update_alive_tracks(const std::vector<cv::KeyPoint>& feats,
                                     const std::vector<core::DescriptorNCC>& desc,
                                     const std::vector<int>& match_index,
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
      desc_curr_[i] = desc[mi];
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
void TrackerBFM::replace_dead_tracks(const std::vector<cv::KeyPoint>& feats_c,
                                     const std::vector<core::DescriptorNCC>& desc,
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
          desc_ref_[i] = prev_unused_desc_[j];
          desc_curr_[i] = desc[mi];
          age_[i] = 1;
          j++;
          break;
        }
        j++;
      }
    }
  }
}

// here we save all the unused current features to be used to replace dead feats in the next frame
void TrackerBFM::save_unused_features(const std::vector<cv::KeyPoint>& feats,
                                      const std::vector<core::DescriptorNCC>& descriptors,
                                      std::vector<bool>& unused_features)

{
  prev_unused_feats_.clear();
  prev_unused_desc_.clear();
  // good thing the best Harris corners are from begining to end
  for(size_t i = 0; i < unused_features.size(); i++) {
    if(unused_features[i] == true) {
      prev_unused_feats_.push_back(feats[i]);
      prev_unused_desc_.push_back(descriptors[i]);
      //std::cout << prev_unused_desc_ << "\n\n";
      //cv::waitKey(0);
    }
  }
}

void TrackerBFM::compute_ncc_descriptors(const cv::Mat& img, const std::vector<cv::KeyPoint>& features,
                                         std::vector<core::DescriptorNCC>& patches)
{
  patches.resize(features.size());
  for(size_t i = 0; i < features.size(); i++) {
    int cx = static_cast<int>(features[i].pt.x);
    int cy = static_cast<int>(features[i].pt.y);

    recon::StereoCosts::compute_ncc_descriptor<PixelType>(img, cx, cy, patch_size_, kPixelTypeOpenCV,
                                                          patches[i]);
  }
}

int TrackerBFM::countFeatures()
{
  return matches_p_.size();
}

int TrackerBFM::countTracked()
{
  int cnt = 0;
  for(size_t i = 0; i < age_.size(); i++) {
    if(age_[i] > 0)
      cnt++;
  }
  return cnt;
}

FeatureData TrackerBFM::getFeatureData(int i)
{
  FeatureData fdata;
  fdata.feat_ = feature(i);
  //fdata.desc_prev_ = desc_prev_[i].vec;
  //fdata.desc_prev_ = desc_ref_[i].vec.reshape(1, cbh_);
  //fdata.desc_curr_ = desc_curr_[i].vec.reshape(1, cbh_);
  desc_ref_[i].vec.reshape(1, patch_size_).convertTo(fdata.desc_prev_, CV_8U);
  desc_curr_[i].vec.reshape(1, patch_size_).convertTo(fdata.desc_curr_, CV_8U);

  //fdata.ncc_prev_ = desc_prev_[i];
  fdata.ncc_prev_ = desc_ref_[i];
  fdata.ncc_curr_ = desc_curr_[i];
  return fdata;
}

void TrackerBFM::printStats()
{
  std::cout << "[TrackerBFM] Active tracks: " << countTracked() << "\n";
  std::cout << "[TrackerBFM] Average track age: " << (double) age_acc_ / death_count_ << "\n";
}

FeatureInfo TrackerBFM::feature(int i)
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


void TrackerBFM::showTrack(int i)
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
}

} // end namespace
